"""
Reddit listing scraper using old.reddit.com static HTML only.
No PRAW, no Reddit API — requests + BeautifulSoup only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .utils import (
    get_headers,
    load_config,
    random_sleep,
    retry_with_backoff,
)

logger = logging.getLogger("visscore_scraper.reddit")


# Direct image URL patterns we accept (no videos, gifs, albums, or articles)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
I_REDDIT_DOMAIN = "i.redd.it"
IMGUR_SINGLE_PATTERN = re.compile(
    r"^https?://(?:i\.)?imgur\.com/[a-zA-Z0-9]+\.(?:jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)
# Reject these
GALLERY_PATTERN = re.compile(r"reddit\.com/gallery/", re.IGNORECASE)
VIDEO_GIF_EXT = (".gif", ".gifv", ".mp4", ".webm", ".mov")


@dataclass
class PostRecord:
    """Metadata for one scraped post."""

    post_id: str
    title: str
    upvotes: int
    image_url: str | None
    permalink: str
    subreddit: str


def _parse_upvotes(text: str | None) -> int:
    """Parse score string like '17.2k', '1.2k', '923' into int. Returns 0 on failure."""
    if not text:
        return 0
    text = text.strip().replace(",", "").lower()
    multiplier = 1
    if text.endswith("k"):
        text = text[:-1].strip()
        multiplier = 1000
    elif text.endswith("m"):
        text = text[:-1].strip()
        multiplier = 1_000_000
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def _is_direct_image_url(url: str | None) -> bool:
    """True if URL is a direct image we are allowed to download."""
    if not url:
        return False
    parsed = urlparse(url)
    path_lower = (parsed.path or "").lower()
    # Reject galleries and video/gif
    if GALLERY_PATTERN.search(url):
        return False
    for ext in VIDEO_GIF_EXT:
        if path_lower.endswith(ext) or ext in path_lower:
            return False
    # i.redd.it
    if I_REDDIT_DOMAIN in (parsed.netloc or ""):
        return any(path_lower.endswith(ext) for ext in IMAGE_EXTENSIONS) or True
    # Direct imgur single image
    if IMGUR_SINGLE_PATTERN.match(url):
        return True
    # Any URL ending with allowed image extension
    if any(path_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return True
    return False


def _normalize_permalink(href: str | None, base_url: str, subreddit: str) -> str:
    """Ensure permalink is absolute and points to old.reddit.com."""
    if not href:
        return ""
    if href.startswith("/"):
        href = urljoin(base_url, href)
    # Prefer old.reddit.com for consistency
    parsed = urlparse(href)
    if "reddit.com" in (parsed.netloc or ""):
        href = f"https://old.reddit.com{parsed.path or ''}"
        if parsed.query:
            href += "?" + parsed.query
    return href


def _extract_post_id(fullname: str | None, permalink: str) -> str:
    """Get post ID from data-fullname (t3_xxx) or from permalink /comments/ID/."""
    if fullname and fullname.startswith("t3_"):
        return fullname[3:].strip()
    match = re.search(r"/comments/([a-zA-Z0-9]+)/?", permalink)
    if match:
        return match.group(1)
    return ""


class RedditScraper:
    """
    Scrapes post listings from old.reddit.com using requests + BeautifulSoup.
    Handles pagination via the 'after' parameter. Does not use Reddit API or PRAW.
    """

    def __init__(
        self,
        base_url: str = "https://old.reddit.com",
        user_agent: str | None = None,
        delay_min: float = 2.0,
        delay_max: float = 5.0,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = get_headers(user_agent)
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _get_url(self, url: str) -> requests.Response:
        """Fetch URL with retry and exponential backoff."""
        def _get() -> requests.Response:
            r = requests.get(url, headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            return r

        return retry_with_backoff(
            _get,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            logger=logger,
            exceptions=(requests.RequestException,),
        )

    def _build_listing_url(
        self,
        subreddit: str,
        sort: str,
        time_filter: str | None,
        limit: int,
        after: str | None,
    ) -> str:
        """Build old.reddit.com listing URL. sort in top/hot/new; time_filter only for top."""
        subreddit = subreddit.strip().lstrip("/r/")
        path = f"/r/{subreddit}/{sort}/"
        params: list[str] = [f"limit={min(limit, 100)}"]
        if sort == "top" and time_filter:
            params.append(f"t={time_filter}")
        if after:
            params.append(f"after={after}")
        return self.base_url + path + "?" + "&".join(params)

    def _parse_listing_page(self, html: str, subreddit: str) -> tuple[list[PostRecord], str | None]:
        """
        Parse one listing page HTML. Returns (list of PostRecord, next_after or None).
        Uses div.thing when present; falls back to other selectors for resilience.
        """
        soup = BeautifulSoup(html, "html.parser")
        posts: list[PostRecord] = []
        next_after: str | None = None

        # Pagination: find "next" link for 'after' token
        next_link = soup.find("a", rel="nofollow", string=re.compile(r"next\s*›", re.I))
        if next_link and next_link.get("href"):
            href = next_link["href"]
            match = re.search(r"after=([a-zA-Z0-9_]+)", href)
            if match:
                next_after = match.group(1)

        # Post blocks: old Reddit uses div.thing with data-fullname
        things = soup.find_all("div", class_="thing")
        if not things:
            # Fallback: look for entry-like blocks (e.g. link listing)
            things = soup.select("div[data-fullname]")

        for thing in things:
            try:
                fullname = thing.get("data-fullname") or ""
                if not fullname.startswith("t3_"):
                    continue
                # Skip promoted
                if "promoted" in (thing.get("class") or []):
                    continue

                # Title link: first a.title or similar
                title_el = thing.find("a", class_="title") or thing.select_one("a[data-event-action=title]")
                if not title_el:
                    continue
                title_text = (title_el.get_text() or "").strip()
                post_url = title_el.get("href") or ""
                if not post_url.startswith("http"):
                    post_url = urljoin(self.base_url, post_url)

                # Permalink: prefer comments link; if title links to comments we have it
                comments_link = thing.find("a", attrs={"data-event-action": "comments"}) or thing.find("a", class_="comments")
                if comments_link and comments_link.get("href"):
                    permalink = urljoin(self.base_url, comments_link["href"])
                else:
                    permalink = _normalize_permalink(post_url, self.base_url, subreddit)

                # Score
                score_el = thing.find("div", class_="score") or thing.find("span", class_="score")
                score_text = (score_el.get_text() if score_el else "") or thing.get("data-score") or ""
                upvotes = _parse_upvotes(score_text) if score_text else 0

                # Image URL: use title link if it's already a direct image; else resolve from permalink
                image_url: str | None = None
                if _is_direct_image_url(post_url):
                    image_url = post_url
                # If title links to comments (not direct image), we skip here; caller may resolve
                # via resolve_image_from_permalink() to get og:image / i.redd.it from the post page.

                post_id = _extract_post_id(fullname, permalink)
                if not post_id:
                    post_id = fullname.replace("t3_", "") if fullname else ""

                record = PostRecord(
                    post_id=post_id,
                    title=title_text,
                    upvotes=upvotes,
                    image_url=image_url,
                    permalink=permalink,
                    subreddit=subreddit,
                )
                posts.append(record)
            except Exception as e:
                logger.debug("Skipping one post block due to parse error: %s", e)
                continue

        return posts, next_after

    def resolve_image_from_permalink(self, permalink: str) -> str | None:
        """
        Fetch the post page and extract the main image URL (og:image or first i.redd.it).
        Returns None on failure or if not a direct image. One extra request per call.
        """
        if not permalink:
            return None
        try:
            resp = self._get_url(permalink)
            soup = BeautifulSoup(resp.text, "html.parser")
            # og:image is the canonical shared image
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                url = (og["content"] or "").strip()
                if _is_direct_image_url(url):
                    return url
            # Fallback: first i.redd.it or single imgur in content
            for a in soup.select("a[href*='i.redd.it'], a[href*='imgur.com']"):
                href = a.get("href") or ""
                if _is_direct_image_url(href):
                    return href
        except Exception as e:
            logger.debug("Could not resolve image from permalink %s: %s", permalink, e)
        return None

    def scrape(
        self,
        subreddit: str,
        sort: str = "top",
        time_filter: str | None = "month",
        max_posts: int = 100,
        resolve_image_from_permalink: bool = True,
    ) -> list[PostRecord]:
        """
        Scrape up to max_posts from the subreddit. Paginates using 'after'.
        sort: top | hot | new. time_filter: day | week | month | year | all (only for top).
        When resolve_image_from_permalink is True, fetches post page for image URL when listing
        only has a comments link (adds one request per such post).
        """
        if sort not in ("top", "hot", "new"):
            sort = "top"
        if sort != "top":
            time_filter = None
        else:
            time_filter = time_filter or "month"

        all_posts: list[PostRecord] = []
        after: str | None = None
        page_limit = 25
        subreddit_norm = subreddit.strip().lstrip("/r/")

        while len(all_posts) < max_posts:
            url = self._build_listing_url(
                subreddit=subreddit_norm,
                sort=sort,
                time_filter=time_filter,
                limit=page_limit,
                after=after,
            )
            try:
                resp = self._get_url(url)
                batch, next_after = self._parse_listing_page(resp.text, subreddit_norm)
            except Exception as e:
                logger.exception("Failed to fetch or parse listing: %s", e)
                break

            for p in batch:
                if len(all_posts) >= max_posts:
                    break
                if p.image_url is None and resolve_image_from_permalink and p.permalink:
                    resolved = self.resolve_image_from_permalink(p.permalink)
                    if resolved:
                        p = PostRecord(
                            post_id=p.post_id,
                            title=p.title,
                            upvotes=p.upvotes,
                            image_url=resolved,
                            permalink=p.permalink,
                            subreddit=p.subreddit,
                        )
                    random_sleep(self.delay_min, self.delay_max)
                if p.image_url is not None:
                    all_posts.append(p)

            if not next_after or not batch:
                break
            after = next_after
            random_sleep(self.delay_min, self.delay_max)

        return all_posts
