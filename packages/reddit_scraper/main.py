#!/usr/bin/env python3
"""
VisScore Reddit chart scraper — production entrypoint.
Uses old.reddit.com + requests + BeautifulSoup only (no PRAW, no Reddit API).

CLI: subreddit, sort, time filter, max_posts. Supports resume and config via config.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root, packages/reddit_scraper, or a clone named vis_scraper
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scraper.reddit_scraper import RedditScraper
from scraper.image_downloader import ImageDownloader
from scraper.utils import load_config, setup_logging, random_sleep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape chart images from old.reddit.com (r/dataisbeautiful, r/dataisugly). No API keys.",
    )
    parser.add_argument(
        "subreddit",
        nargs="?",
        default="dataisbeautiful",
        help="Subreddit name without r/ (default: dataisbeautiful)",
    )
    parser.add_argument(
        "--sort",
        choices=["top", "hot", "new"],
        default="top",
        help="Listing sort (default: top)",
    )
    parser.add_argument(
        "--time",
        dest="time_filter",
        choices=["day", "week", "month", "year", "all"],
        default="month",
        help="Time filter for 'top' only (default: month)",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=50,
        metavar="N",
        help="Max number of image posts to process (default: 50)",
    )
    parser.add_argument(
        "--no-resolve",
        action="store_true",
        help="Do not fetch post page to resolve image URL when listing has comments link",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume: re-download and overwrite existing",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: package config, or cwd, or packages/reddit_scraper/)",
    )
    return parser.parse_args()


def _load_existing_metadata(metadata_dir: Path, subreddit: str) -> tuple[set[str], set[str]]:
    """Load existing post_ids and SHA256 hashes from metadata JSON for resume."""
    seen_ids: set[str] = set()
    seen_sha256: set[str] = set()
    pattern = metadata_dir / f"{subreddit}_*.json"
    for path in metadata_dir.glob(f"{subreddit}_*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if item.get("post_id"):
                            seen_ids.add(item["post_id"])
                        if item.get("sha256"):
                            seen_sha256.add(item["sha256"])
            elif isinstance(data, dict) and "posts" in data:
                for item in data["posts"]:
                    if isinstance(item, dict):
                        if item.get("post_id"):
                            seen_ids.add(item["post_id"])
                        if item.get("sha256"):
                            seen_sha256.add(item["sha256"])
        except (json.JSONDecodeError, OSError):
            continue
    return seen_ids, seen_sha256


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    scraper_cfg = config.get("scraper") or {}
    image_cfg = config.get("image") or {}
    paths_cfg = config.get("paths") or {}
    resume = config.get("resume", True) and not args.no_resume

    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / (paths_cfg.get("raw_images_dir") or "data/raw")
    metadata_dir = base_dir / (paths_cfg.get("metadata_dir") or "data/metadata")
    logs_dir = base_dir / (paths_cfg.get("logs_dir") or "logs")

    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_dir)
    subreddit = args.subreddit.strip().lstrip("/r/")
    out_images = raw_dir / subreddit
    out_images.mkdir(parents=True, exist_ok=True)

    user_agent = scraper_cfg.get("user_agent") or "visscore-research-scraper/1.0 (academic project)"
    delay_min = float(scraper_cfg.get("request_delay_min_seconds", 2))
    delay_max = float(scraper_cfg.get("request_delay_max_seconds", 5))
    timeout = int(scraper_cfg.get("timeout_seconds", 30))
    max_retries = int(scraper_cfg.get("max_retries", 3))
    backoff = float(scraper_cfg.get("retry_backoff_base_seconds", 2))

    min_w = int(image_cfg.get("min_width_px", 300))
    min_h = int(image_cfg.get("min_height_px", 300))

    seen_ids: set[str] = set()
    seen_sha256: set[str] = set()
    if resume:
        seen_ids, seen_sha256 = _load_existing_metadata(metadata_dir, subreddit)
        # Also consider existing image files as done
        for f in out_images.iterdir():
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                stem = f.stem
                if stem and stem not in seen_ids:
                    seen_ids.add(stem)
        if seen_ids or seen_sha256:
            logger.info("Resume: skipping %s existing post(s), %s known hash(es)", len(seen_ids), len(seen_sha256))

    scraper = RedditScraper(
        base_url=scraper_cfg.get("base_url", "https://old.reddit.com"),
        user_agent=user_agent,
        delay_min=delay_min,
        delay_max=delay_max,
        timeout=timeout,
        max_retries=max_retries,
        backoff_base=backoff,
    )

    logger.info("Scraping r/%s sort=%s time=%s max_posts=%s", subreddit, args.sort, args.time_filter, args.max_posts)
    posts = scraper.scrape(
        subreddit=subreddit,
        sort=args.sort,
        time_filter=args.time_filter if args.sort == "top" else None,
        max_posts=args.max_posts,
        resolve_image_from_permalink=not args.no_resolve,
    )
    logger.info("Got %s posts with image URLs", len(posts))

    downloader = ImageDownloader(
        output_dir=out_images,
        min_width=min_w,
        min_height=min_h,
        user_agent=user_agent,
        timeout=timeout,
        max_retries=max_retries,
        backoff_base=backoff,
    )

    metadata_list: list[dict] = []
    for i, post in enumerate(posts):
        if resume and post.post_id in seen_ids:
            logger.debug("Resume: skip existing post %s", post.post_id)
            continue
        random_sleep(delay_min, delay_max)
        meta = downloader.download_and_validate(post, seen_sha256=seen_sha256)
        if meta:
            metadata_list.append(meta)
            seen_ids.add(post.post_id)

    metadata_path = metadata_dir / f"{subreddit}_{args.sort}_{args.time_filter}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    logger.info("Saved metadata for %s images to %s", len(metadata_list), metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
