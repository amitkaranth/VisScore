"""
Shared utilities: config loading, logging, retry with exponential backoff,
randomized sleep, and request headers.
No Reddit API or PRAW — scraper uses only requests + BeautifulSoup on old.reddit.com.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

T = TypeVar("T")

# Reddit blocks generic or bot-like User-Agents (403). Use a real browser string.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load YAML config. Searches vis_scraper directory and cwd.
    Returns dict; missing keys should be handled by callers with defaults.
    """
    if config_path is None:
        base = Path(__file__).resolve().parent.parent
        candidates = [
            base / "config.yaml",
            Path.cwd() / "config.yaml",
            Path.cwd() / "vis_scraper" / "config.yaml",
        ]
        for p in candidates:
            if p.is_file():
                config_path = p
                break
        else:
            return {}

    path = Path(config_path)
    if not path.is_file():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def setup_logging(
    logs_dir: str | Path,
    name: str = "visscore_scraper",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure file and console logging. Creates logs_dir if needed.
    Returns the logger instance for the given name.
    """
    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    fh = logging.FileHandler(log_dir / "scraper.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def random_sleep(min_seconds: float, max_seconds: float) -> None:
    """
    Sleep for a random duration between min_seconds and max_seconds.
    Used between requests to avoid rate limits and be respectful to the host.
    """
    duration = random.uniform(min_seconds, max_seconds)
    time.sleep(duration)


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int,
    backoff_base: float,
    logger: logging.Logger | None = None,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """
    Execute fn(); on exception, retry with exponential backoff.
    Raises the last exception if all retries fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except exceptions as e:
            last_exc = e
            if attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                if logger:
                    logger.warning(
                        "Attempt %s/%s failed: %s. Retrying in %.1fs.",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        delay,
                    )
                time.sleep(delay)
            else:
                if logger:
                    logger.error("All %s attempts failed.", max_retries + 1)
                raise
    raise last_exc  # type: ignore[misc]


def get_headers(user_agent: str | None = None) -> dict[str, str]:
    """Build request headers. Reddit returns 403 without a browser-like User-Agent."""
    ua = user_agent or DEFAULT_USER_AGENT
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "DNT": "1",
    }


def get_image_headers(user_agent: str | None = None) -> dict[str, str]:
    """
    Headers for requesting image URLs. Reddit returns HTML (cookie page) instead of
    image bytes when we use document-style headers; image-style headers fix that.
    """
    ua = user_agent or DEFAULT_USER_AGENT
    return {
        "User-Agent": ua,
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://old.reddit.com/",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
    }
