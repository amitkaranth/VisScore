"""Seaborn-based charts with heavy styling (BI-tool adjacent clutter)."""

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


def chart_sns_heatmap_loud(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="whitegrid", font_scale=float(rng.uniform(1.0, 1.2)))
    r, c = int(rng.integers(6, 14)), int(rng.integers(6, 14))
    m = rng.uniform(-1, 1, (r, c))
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="#f5f5f5")
    cmap = str(rng.choice(["rainbow", "turbo", "gist_ncar", "hsv"]))
    sns.heatmap(
        m,
        ax=ax,
        cmap=cmap,
        annot=rng.random() < 0.55,
        fmt=".1f" if r * c < 120 else ".0f",
        linewidths=float(rng.uniform(0.4, 1.2)),
        linecolor=str(rng.choice(["gray", "white", "black"])),
        cbar=True,
    )
    ax.set_title("Heat Matrix", fontsize=int(rng.integers(13, 17)), fontweight="bold")
    fig.tight_layout(pad=0.4)
    return fig, {"chart_family": "sns_heatmap_loud", "rows": r, "cols": c, "renderer": "seaborn"}


def chart_sns_box_swarm_busy(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="whitegrid")
    cats = int(rng.integers(6, 14))
    labels = [f"Q{i}" for i in range(cats)]
    shifts = {lab: float(rng.normal(0, 0.25)) for lab in labels}
    n = int(rng.integers(12, 28)) * cats
    x = rng.choice(labels, n)
    y = np.array([shifts[xi] + rng.normal(0, 1) for xi in x], dtype=float)
    df = pd.DataFrame({"x": x, "y": y})
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    pal = str(rng.choice(["Set2", "Set3", "pastel", "bright"]))
    sns.boxplot(data=df, x="x", y="y", ax=ax, palette=pal, linewidth=float(rng.uniform(1.2, 2.2)))
    sns.stripplot(data=df, x="x", y="y", ax=ax, color="black", alpha=0.35, size=float(rng.uniform(2.5, 4.5)))
    ax.grid(True, which="both", alpha=0.6)
    ax.set_title("KPI Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout(pad=0.45)
    return fig, {"chart_family": "sns_box_swarm_busy", "n_categories": cats, "renderer": "seaborn"}


def chart_sns_kde_overlay_loud(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="#eaeaea")
    layers = int(rng.integers(3, 7))
    for i in range(layers):
        d = rng.normal(float(i) * 0.4, float(rng.uniform(0.7, 1.2)), int(rng.integers(500, 1500)))
        sns.kdeplot(data=d, ax=ax, fill=True, alpha=float(rng.uniform(0.25, 0.5)), linewidth=2.5)
    ax.set_title("Density Stack", fontsize=15, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return fig, {"chart_family": "sns_kde_overlay_loud", "n_layers": layers, "renderer": "seaborn"}


def chart_sns_bar_estimator_show(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    cats = [f"R{i}" for i in range(int(rng.integers(8, 16)))]
    df = pd.DataFrame({"c": rng.choice(cats, 400), "v": rng.uniform(0, 1, 400)})
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    sns.barplot(data=df, x="c", y="v", ax=ax, palette="rocket", errorbar="sd", capsize=0.08, err_kws={"linewidth": 2})
    ax.tick_params(axis="x", rotation=40)
    ax.set_title("Regional Averages ± spread", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.7)
    fig.tight_layout(pad=0.45)
    return fig, {"chart_family": "sns_bar_estimator_show", "n_categories": len(cats), "renderer": "seaborn"}


def chart_sns_scatter_hue_size(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="whitegrid")
    n = int(rng.integers(80, 220))
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 1, n),
            "y": rng.uniform(0, 1, n),
            "cat": rng.choice(list("ABCDEFG"), n),
            "s": rng.uniform(20, 400, n),
        }
    )
    fig, ax = plt.subplots(figsize=_figsize(w_px, h_px, dpi), dpi=dpi, facecolor="#f8f8ff")
    sns.scatterplot(data=df, x="x", y="y", hue="cat", size="s", sizes=(30, 300), ax=ax, palette="tab10", alpha=0.85)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=True, title="Groups")
    ax.set_title("Multi-series Bubble-ish", fontsize=14, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return fig, {"chart_family": "sns_scatter_hue_size", "n_points": n, "renderer": "seaborn"}


def chart_sns_clustermap_busy(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    sns.set_theme(style="white")
    r, c = int(rng.integers(8, 16)), int(rng.integers(8, 16))
    data = pd.DataFrame(rng.uniform(0, 1, (r, c)))
    cg = sns.clustermap(
        data,
        figsize=_figsize(w_px, h_px, dpi),
        cmap=str(rng.choice(["coolwarm", "viridis", "cubehelix"])),
        linewidths=float(rng.uniform(0.2, 0.8)),
        dendrogram_ratio=float(rng.uniform(0.08, 0.15)),
        cbar_pos=(0.02, 0.82, 0.03, 0.12),
    )
    cg.ax_row_dendrogram.set_visible(rng.random() < 0.85)
    cg.ax_col_dendrogram.set_visible(rng.random() < 0.85)
    fig = cg.fig
    fig.patch.set_facecolor("#f0f0f0")
    return fig, {"chart_family": "sns_clustermap_busy", "rows": r, "cols": c, "renderer": "seaborn"}
