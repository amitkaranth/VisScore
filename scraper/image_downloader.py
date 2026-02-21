"""
Image download with HTTP checks and file validation (PIL + SHA256).
When the server returns HTML instead of image bytes, we parse it and extract the real image URL.
Uses requests only; no Reddit API.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from .reddit_scraper import PostRecord
from .utils import get_image_headers, retry_with_backoff
from .validators import validate_content_type, validate_and_hash

logger = logging.getLogger("visscore_scraper.downloader")

# Direct image URL pattern for extraction from HTML
REDDIT_IMAGE_URL_RE = re.compile(
    r"https?://(?:i|preview)\.redd\.it/[^\s\"'<>]+(?:\.(?:jpg|jpeg|png|webp))?(?:\?[^\s\"'<>]*)?",
    re.IGNORECASE,
)
def _safe_filename(post_id: str, url: str) -> str:
    """Generate a safe filename: post_id + extension from URL or default .jpg."""
    ext = ".jpg"
    path = (url or "").split("?")[0].lower()
    for e in (".png", ".webp", ".jpeg", ".jpg"):
        if e in path or path.endswith(e):
            ext = e
            break
    return f"{post_id}{ext}"


def _reddit_image_url_to_direct(url: str) -> str:
    """Use i.redd.it for raw image bytes; strip query params that can cause HTML response."""
    url = url.split("?")[0].strip()
    if "preview.redd.it" in url:
        url = url.replace("preview.redd.it", "i.redd.it", 1)
    return url


def _extract_image_url_from_html(file_path: Path) -> str | None:
    """
    If the downloaded file is HTML (e.g. Reddit preview page), parse it and return
    the first direct image URL. We prefer i.redd.it over preview.redd.it so the
    next request returns raw image bytes.
    """
    try:
        raw = file_path.read_bytes()
    except OSError:
        return None
    head = raw[:8192].decode("utf-8", errors="ignore")
    if "<!DOCTYPE" not in head and "<html" not in head and "<meta" not in head:
        return None
    soup = BeautifulSoup(raw, "html.parser")
    candidates: list[str] = []
    # 1. og:image
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        url = (og["content"] or "").strip()
        if "redd.it" in url:
            candidates.append(url)
    # 2. img src with redd.it
    for img in soup.find_all("img", src=True):
        src = (img.get("src") or "").strip()
        m = REDDIT_IMAGE_URL_RE.search(src)
        if m:
            candidates.append(m.group(0))
    # 3. Any link matching redd.it image
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        m = REDDIT_IMAGE_URL_RE.search(href)
        if m:
            candidates.append(m.group(0))
    if not candidates:
        full_text = " ".join(str(t) for t in soup.find_all(string=True))
        m = REDDIT_IMAGE_URL_RE.search(full_text)
        if m:
            candidates.append(m.group(0))
    if not candidates:
        return None
    # Prefer i.redd.it so we get raw image; rewrite preview -> i
    best = candidates[0]
    return _reddit_image_url_to_direct(best)


class ImageDownloader:
    """
    Downloads images from URLs, validates HTTP and Content-Type, then validates
    file with PIL (dimensions, corruption) and computes SHA256 for deduplication.
    """

    def __init__(
        self,
        output_dir: str | Path,
        min_width: int = 300,
        min_height: int = 300,
        user_agent: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_width = min_width
        self.min_height = min_height
        self.headers = get_image_headers(user_agent)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _download_to_path(self, url: str, filename: str | None = None) -> tuple[Path | None, str | None]:
        """
        Download URL to output_dir. Returns (path, error_message).
        Verifies status code only. Content-Type is not required to be image/* because
        many hosts (e.g. i.redd.it) often send text/html or omit it; we rely on PIL
        to reject non-image content after download.
        """
        def _get() -> requests.Response:
            r = requests.get(url, headers=self.headers, timeout=self.timeout, stream=True)
            r.raise_for_status()
            return r

        try:
            resp = retry_with_backoff(
                _get,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
                logger=logger,
                exceptions=(requests.RequestException,),
            )
        except Exception as e:
            return None, str(e)

        out_name = filename or _safe_filename("tmp", url)
        out_path = self.output_dir / out_name
        try:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            return None, str(e)

        # Optional log when server sent wrong Content-Type; we still validate with PIL
        if not validate_content_type(resp.headers.get("Content-Type")):
            logger.debug(
                "Content-Type was %s for %s; validating with PIL",
                resp.headers.get("Content-Type") or "(none)",
                filename or url[:50],
            )

        return out_path, None

    def download_and_validate(
        self,
        record: PostRecord,
        seen_sha256: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Download image for this post, validate dimensions and integrity, optionally
        dedupe by SHA256. If the server returns HTML instead of image, we parse it
        and retry with the extracted image URL. On success returns metadata dict; else None.
        """
        if not record.image_url:
            logger.debug("Skipping post %s: no image URL", record.post_id)
            return None

        filename = _safe_filename(record.post_id, record.image_url)
        url_to_try: str | None = record.image_url
        attempt = 0
        max_attempts = 2  # initial URL + one retry from HTML

        while attempt < max_attempts and url_to_try:
            attempt += 1
            path, err = self._download_to_path(url_to_try, filename=filename)
            if err:
                logger.warning("Download failed for %s: %s", record.post_id, err)
                return None
            if path is None:
                return None

            try:
                result = validate_and_hash(
                    path,
                    min_width=self.min_width,
                    min_height=self.min_height,
                )
            except Exception as e:
                # Often "cannot identify image file" = server returned HTML
                fallback_url = _extract_image_url_from_html(path)
                path.unlink(missing_ok=True)
                if fallback_url and fallback_url != url_to_try:
                    logger.debug("Got non-image response for %s, retrying with %s", record.post_id, fallback_url[:60])
                    url_to_try = fallback_url
                    continue
                logger.warning("Validation failed for %s: %s", record.post_id, e)
                return None

            if not result.valid:
                # Dimensions too small or other PIL rejection; no retry (file was a valid image)
                path.unlink(missing_ok=True)
                logger.info(
                    "Skipping post %s: %s",
                    record.post_id,
                    result.reason or "validation failed",
                )
                return None

            # Valid image
            if seen_sha256 is not None and result.sha256 and result.sha256 in seen_sha256:
                logger.info("Skipping duplicate (SHA256) post %s", record.post_id)
                path.unlink(missing_ok=True)
                return None

            if result.sha256 and seen_sha256 is not None:
                seen_sha256.add(result.sha256)

            meta = {
                "post_id": record.post_id,
                "title": record.title,
                "upvotes": record.upvotes,
                "image_url": url_to_try or record.image_url,
                "permalink": record.permalink,
                "subreddit": record.subreddit,
                "local_path": str(path),
                "width": result.width,
                "height": result.height,
                "sha256": result.sha256,
            }
            logger.info("Downloaded and validated: %s -> %s", record.post_id, path.name)
            return meta

        return None
