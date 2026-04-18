"""Lightweight PIL-based augmentation (optional)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def augment_image(
    image: Image.Image,
    rng: np.random.Generator,
    strength: float,
) -> Image.Image:
    """
    strength in [0, 1]. Keeps class semantics; mild JPEG, blur, brightness/contrast jitter.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return image
    img = image.convert("RGB")
    if rng.random() < 0.55:
        br = 1.0 + (rng.random() * 2 - 1) * 0.22 * strength
        img = ImageEnhance.Brightness(img).enhance(br)
    if rng.random() < 0.55:
        ct = 1.0 + (rng.random() * 2 - 1) * 0.25 * strength
        img = ImageEnhance.Contrast(img).enhance(ct)
    if rng.random() < 0.45:
        radius = float(rng.uniform(0.15, 0.9) * strength)
        if radius > 0.05:
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        q = int(92 - 35 * strength)
        q = max(55, min(95, q))
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img
