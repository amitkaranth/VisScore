"""
Image validation for the dataset pipeline.
Verifies HTTP response, Content-Type, dimensions (PIL), and computes SHA256 for deduplication.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import NamedTuple

from PIL import Image

logger = logging.getLogger("visscore_scraper.validators")


class ValidationResult(NamedTuple):
    """Result of validating a downloaded image."""

    valid: bool
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    reason: str | None = None


def validate_content_type(content_type: str | None, prefix: str = "image/") -> bool:
    """
    Verify that the response Content-Type indicates an image.
    Reddit and CDNs may return e.g. 'image/jpeg' or 'image/png'.
    """
    if not content_type:
        return False
    return content_type.strip().lower().startswith(prefix)


def validate_image_file(
    file_path: Path,
    min_width: int = 300,
    min_height: int = 300,
) -> ValidationResult:
    """
    Open image with PIL; reject if corrupted, or dimensions below minimum.
    Returns ValidationResult with valid flag, dimensions, and optional reason.
    """
    try:
        with Image.open(file_path) as img:
            img.load()
            w, h = img.size
    except Exception as e:
        logger.debug("Rejected corrupted or invalid image %s: %s", file_path, e)
        return ValidationResult(
            valid=False,
            reason=f"corrupted or invalid image: {e!s}",
        )

    if w < min_width or h < min_height:
        return ValidationResult(
            valid=False,
            width=w,
            height=h,
            reason=f"dimensions {w}x{h} below minimum {min_width}x{min_height}",
        )

    return ValidationResult(valid=True, width=w, height=h)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file for deduplication. Reads in chunks for large files."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_and_hash(
    file_path: Path,
    min_width: int = 300,
    min_height: int = 300,
) -> ValidationResult:
    """
    Full validation: PIL dimensions check plus SHA256.
    Attach sha256 to the result for downstream dedup even when valid=True.
    """
    result = validate_image_file(file_path, min_width=min_width, min_height=min_height)
    if not result.valid:
        return result
    digest = compute_sha256(file_path)
    return ValidationResult(
        valid=True,
        width=result.width,
        height=result.height,
        sha256=digest,
    )
