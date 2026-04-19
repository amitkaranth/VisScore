"""Tufte-aligned chart builders (matplotlib, Agg)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _figsize_inches(w_px: int, h_px: int, dpi: float) -> tuple[float, float]:
    return (max(w_px / dpi, 1.0), max(h_px / dpi, 1.0))


def _tufte_palette(rng: np.random.Generator) -> tuple[str, str]:
    base = str(rng.choice(["#2a2a2a", "#333333", "#3d4f5f", "#2f3d2f"]))
    accent = str(rng.choice(["#4a6fa5", "#5a7d6a", "#8b7355", "#6b5b73", "#4d6b8a"]))
    return base, accent


def _style_tufte_axes(ax: Any, rng: np.random.Generator) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lw = float(rng.uniform(0.45, 0.75))
    for side in ("left", "bottom"):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color("#444444")
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=int(rng.integers(7, 10)),
        width=lw,
        length=int(rng.integers(2, 4)),
        colors="#444444",
    )
    ax.grid(False)


def chart_tufte_line(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    n = int(rng.integers(24, 140))
    x = np.linspace(0, 1, n)
    y = np.cumsum(rng.normal(0, 0.07, n))
    y -= y.mean()
    base, _ = _tufte_palette(rng)
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    ax.plot(x, y, color=base, linewidth=float(rng.uniform(0.9, 1.5)), solid_capstyle="butt")
    ax.set_xlim(0, 1)
    _style_tufte_axes(ax, rng)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "line", "n_points": n}


def chart_tufte_scatter(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    n = int(rng.integers(40, 220))
    x = rng.uniform(0, 1, n)
    y = 0.35 * x + rng.normal(0, 0.12, n)
    base, accent = _tufte_palette(rng)
    color = accent if rng.random() < 0.5 else base
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    s = float(rng.uniform(8, 22))
    ax.scatter(x, y, s=s, c=color, alpha=float(rng.uniform(0.55, 0.85)), linewidths=0)
    _style_tufte_axes(ax, rng)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "scatter", "n_points": n}


def chart_tufte_bar_horizontal(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    k = int(rng.integers(5, 14))
    y = np.arange(k)
    w = rng.uniform(0.2, 1.0, k)
    _, accent = _tufte_palette(rng)
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    ax.barh(y, w, height=float(rng.uniform(0.55, 0.78)), color=accent, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}" for i in range(k)], fontsize=int(rng.integers(7, 9)))
    _style_tufte_axes(ax, rng)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "bar_horizontal", "n_bars": k}


def chart_tufte_dot_strip(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    groups = int(rng.integers(4, 10))
    pts_per = int(rng.integers(6, 18))
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    base, _ = _tufte_palette(rng)
    for g in range(groups):
        x = rng.normal(g, 0.12, pts_per)
        y = np.full(pts_per, g, dtype=float) + rng.normal(0, 0.04, pts_per)
        ax.scatter(x, y, s=float(rng.uniform(10, 20)), c=base, alpha=0.75, linewidths=0)
    ax.set_yticks(range(groups))
    ax.set_yticklabels([f"G{j+1}" for j in range(groups)], fontsize=int(rng.integers(7, 9)))
    _style_tufte_axes(ax, rng)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "dot_strip", "n_groups": groups, "pts_per_group": pts_per}


def chart_tufte_small_multiples(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    rows, cols = 2, 2
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=_figsize_inches(w_px, h_px, dpi),
        dpi=dpi,
        facecolor="white",
        sharex=True,
    )
    base, _ = _tufte_palette(rng)
    for ax in axes.ravel():
        n = int(rng.integers(18, 60))
        x = np.linspace(0, 1, n)
        y = np.cumsum(rng.normal(0, 0.05, n))
        ax.plot(x, y, color=base, linewidth=float(rng.uniform(0.7, 1.1)))
        _style_tufte_axes(ax, rng)
    fig.tight_layout(pad=0.45)
    fig.subplots_adjust(wspace=0.35, hspace=0.35)
    return fig, {"chart_family": "small_multiples", "rows": rows, "cols": cols}


def chart_tufte_box(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    k = int(rng.integers(2, 6))
    data = [rng.normal(0, 1, int(rng.integers(40, 120))).tolist() for _ in range(k)]
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        widths=float(rng.uniform(0.35, 0.55)),
        medianprops={"color": "#222222", "linewidth": 1.0},
        boxprops={"facecolor": "#e8e8e8", "edgecolor": "#555555", "linewidth": 0.6},
        whiskerprops={"color": "#555555", "linewidth": 0.6},
        capprops={"color": "#555555", "linewidth": 0.6},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    _style_tufte_axes(ax, rng)
    ax.set_xticklabels([f"S{i+1}" for i in range(k)], fontsize=int(rng.integers(7, 9)))
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "box", "n_series": k}


def chart_tufte_sparkline(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    n = int(rng.integers(48, 160))
    x = np.arange(n)
    y = np.cumsum(rng.normal(0, 0.04, n))
    base, _ = _tufte_palette(rng)
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    ax.plot(x, y, color=base, linewidth=float(rng.uniform(1.0, 1.6)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(x=0.02, y=0.15)
    fig.tight_layout(pad=0.15)
    return fig, {"chart_family": "sparkline", "n_points": n}


def chart_tufte_area(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    n = int(rng.integers(30, 100))
    x = np.linspace(0, 1, n)
    y = np.abs(np.cumsum(rng.normal(0, 0.06, n)))
    y = y / (y.max() + 1e-9) * float(rng.uniform(0.5, 1.0))
    base, accent = _tufte_palette(rng)
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    ax.fill_between(x, y, color=accent, alpha=float(rng.uniform(0.18, 0.32)))
    ax.plot(x, y, color=base, linewidth=float(rng.uniform(0.8, 1.2)))
    _style_tufte_axes(ax, rng)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "area", "n_points": n}
