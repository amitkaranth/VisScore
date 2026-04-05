"""Matplotlib style contexts for digital-looking variation (no print/perspective)."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Styles common on screens / exports; subset may be missing on minimal installs.
_LIGHT_CANDIDATES = [
    "default",
    "ggplot",
    "fast",
    "bmh",
    "Solarize_Light2",
    "seaborn-v0_8-white",
    "seaborn-v0_8-ticks",
    "grayscale",
]

_EXTRA_NON_TUFE_CANDIDATES = [
    "dark_background",
    "seaborn-v0_8-darkgrid",
    "seaborn-v0_8-whitegrid",
]


def resolve_mpl_style_pool(class_label: str, mode: str) -> list[str]:
    """
    mode: none | light | extended (extended adds dark/grid styles for non_tufte only).
    """
    if mode == "none":
        return []
    avail = set(plt.style.available)
    pool = [s for s in _LIGHT_CANDIDATES if s in avail]
    if mode == "extended" and class_label == "non_tufte":
        pool = pool + [s for s in _EXTRA_NON_TUFE_CANDIDATES if s in avail]
    return pool if pool else ["default"]


def is_seaborn_chart_key(chart_key: str) -> bool:
    return chart_key.startswith("sns_")
