"""Figure rendering to PNG and optional PIL augmentation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from visscore_synthetic.augment import augment_image


def figure_to_pil(fig: Figure, dpi: float) -> Image.Image:
    """Rasterize figure to RGB PIL image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_chart_image(
    fig: Figure,
    path: Path,
    dpi: float,
    do_augment: bool,
    aug_rng: np.random.Generator,
    style_strength: float,
) -> dict[str, Any]:
    """Save figure to PNG; optionally augment. Returns extra metadata fields."""
    img = figure_to_pil(fig, dpi)
    out: dict[str, Any] = {"augmented": False, "style_strength": style_strength if do_augment else 0.0}
    if do_augment:
        img = augment_image(img, aug_rng, style_strength)
        out["augmented"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    out["width"], out["height"] = img.size
    return out
