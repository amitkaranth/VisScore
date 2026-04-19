"""Seaborn-based charts with Tufte-like restraint (still statistical / digital)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def _figsize(w_px: int, h_px: int, dpi: float) -> tuple[float, float]:
    return (max(w_px / dpi, 1.0), max(h_px / dpi, 1.0))


def chart_sns_reg_minimal(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="white", font_scale=float(rng.uniform(0.85, 1.0)))
    n = int(rng.integers(45, 160))
    x = rng.uniform(0, 1, n)
    y = float(rng.uniform(0.2, 0.55)) * x + rng.normal(0, float(rng.uniform(0.06, 0.12)), n)
    df = pd.DataFrame({"x": x, "y": y})
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    sns.regplot(
        data=df,
        x="x",
        y="y",
        ax=ax,
        scatter_kws={"s": float(rng.uniform(12, 22)), "alpha": 0.55, "color": "#4a4a4a", "linewidths": 0},
        line_kws={"color": "#222222", "linewidth": float(rng.uniform(1.0, 1.5))},
        ci=None,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "sns_reg_minimal", "n_points": n, "renderer": "seaborn"}


def chart_sns_kde_1d(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="white")
    n = int(rng.integers(800, 2500))
    a = rng.normal(0, 1, n)
    b = rng.normal(2.5, 0.8, int(n * 0.45))
    data = np.concatenate([a, b])
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    color = str(rng.choice(["#3d4f5f", "#4a5a4a", "#5a5a7a"]))
    sns.kdeplot(data=data, ax=ax, fill=True, color=color, alpha=float(rng.uniform(0.2, 0.38)), linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.grid(False)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "sns_kde_1d", "n_points": len(data), "renderer": "seaborn"}


def chart_sns_violin_light(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="white")
    cats = int(rng.integers(4, 9))
    n_each = int(rng.integers(35, 90))
    rows = []
    for c in range(cats):
        rows.extend([(f"C{c+1}", float(v)) for v in rng.normal(float(c) * 0.15, 1, n_each)])
    df = pd.DataFrame(rows, columns=["cat", "val"])
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    sns.violinplot(
        data=df,
        x="cat",
        y="val",
        ax=ax,
        inner="box",
        linewidth=0.8,
        palette="muted",
        saturation=float(rng.uniform(0.35, 0.55)),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "sns_violin_light", "n_categories": cats, "renderer": "seaborn"}


def chart_sns_heatmap_muted(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="white")
    r, c = int(rng.integers(5, 12)), int(rng.integers(5, 12))
    m = rng.normal(0, 1, (r, c))
    m = (m - m.mean()) / (m.std() + 1e-9)
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    cmap = str(rng.choice(["crest", "mako", "vlag"]))
    sns.heatmap(
        m,
        ax=ax,
        cmap=cmap,
        center=0,
        linewidths=0,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    fig.tight_layout(pad=0.25)
    return fig, {"chart_family": "sns_heatmap_muted", "rows": r, "cols": c, "renderer": "seaborn"}


def chart_sns_line_facet_subtle(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="white")
    n = int(rng.integers(25, 70))
    t = np.linspace(0, 1, n)
    g = int(rng.integers(2, 5))
    dfs = []
    for i in range(g):
        y = np.cumsum(rng.normal(0, 0.04, n)) + i * 0.08
        dfs.append(pd.DataFrame({"t": t, "y": y, "g": f"S{i+1}"}))
    df = pd.concat(dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    sns.lineplot(
        data=df,
        x="t",
        y="y",
        hue="g",
        ax=ax,
        legend=False,
        linewidth=float(rng.uniform(1.0, 1.6)),
        palette="dark",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout(pad=0.35)
    return fig, {"chart_family": "sns_line_facet_subtle", "n_series": g, "n_points": n, "renderer": "seaborn"}
