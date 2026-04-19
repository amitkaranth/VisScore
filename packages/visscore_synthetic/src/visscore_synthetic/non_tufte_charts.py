"""Non-Tufte / chartjunk chart builders (matplotlib, Agg)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


def _figsize_inches(w_px: int, h_px: int, dpi: float) -> tuple[float, float]:
    return (max(w_px / dpi, 1.0), max(h_px / dpi, 1.0))


def _loud_colors(rng: np.random.Generator, n: int) -> list[tuple[float, float, float, float]]:
    cmap = rng.choice([cm.rainbow, cm.jet, cm.hsv, cm.cool])
    return [cmap(float(i) / max(n - 1, 1)) for i in range(n)]


def _junk_axes(ax: Any, rng: np.random.Generator) -> None:
    ax.set_facecolor(str(rng.choice(["#f0f0f8", "#fff5e6", "#e8ffe8", "#ffe8f0"])))
    ax.grid(
        True,
        which="both",
        linestyle=str(rng.choice(["-", "--", ":"])),
        linewidth=float(rng.uniform(0.8, 1.8)),
        alpha=float(rng.uniform(0.5, 0.95)),
        color=str(rng.choice(["#8888cc", "#cc8888", "#88cccc"])),
    )
    for s in ax.spines.values():
        s.set_linewidth(float(rng.uniform(2.0, 4.5)))
        s.set_edgecolor(str(rng.choice(["#ff00aa", "#00aa88", "#aa5500", "#4444ff"])))


def chart_non_tufte_bar_rainbow(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    k = int(rng.integers(6, 18))
    x = np.arange(k)
    h = rng.uniform(0.15, 1.0, k)
    colors = _loud_colors(rng, k)
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="#eaeaf8")
    ax.bar(
        x,
        h,
        color=colors,
        edgecolor="#222222",
        linewidth=float(rng.uniform(1.2, 2.8)),
    )
    _junk_axes(ax, rng)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in range(k)], rotation=int(rng.integers(25, 55)))
    ax.legend(["Series A (fake)"], loc="upper right", frameon=True, fancybox=True, shadow=True)
    fig.tight_layout(pad=0.5)
    return fig, {"chart_family": "bar_rainbow", "n_bars": k}


def chart_non_tufte_line_clutter(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    n = int(rng.integers(35, 90))
    x = np.linspace(0, 1, n)
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    series = int(rng.integers(4, 9))
    for s in range(series):
        y = np.cumsum(rng.normal(0, 0.06, n)) + s * 0.15
        c = _loud_colors(rng, series)[s]
        ax.plot(
            x,
            y,
            color=c[:3],
            linewidth=float(rng.uniform(2.0, 4.0)),
            marker=str(rng.choice(["o", "s", "^", "D"])),
            markersize=float(rng.uniform(5, 11)),
        )
    _junk_axes(ax, rng)
    ax.legend([f"K{i}" for i in range(series)], loc="best", ncol=2, framealpha=0.9, shadow=True)
    ax.text(
        0.5,
        0.95,
        "KEY METRICS",
        transform=ax.transAxes,
        ha="center",
        fontsize=int(rng.integers(14, 20)),
        fontweight="bold",
        color="#aa0088",
    )
    fig.tight_layout(pad=0.5)
    return fig, {"chart_family": "line_clutter", "n_series": series, "n_points": n}


def chart_non_tufte_pie_exploded(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    k = int(rng.integers(6, 12))
    sizes = rng.uniform(0.5, 1.0, k)
    sizes = sizes / sizes.sum()
    explode = tuple(float(rng.uniform(0.02, 0.12)) for _ in range(k))
    colors = _loud_colors(rng, k)
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="#fffacd")
    ax.pie(
        sizes,
        explode=explode,
        colors=[c[:3] for c in colors],
        shadow=True,
        autopct="%1.1f%%",
        startangle=float(rng.uniform(0, 90)),
        textprops={"fontsize": int(rng.integers(8, 11))},
    )
    ax.legend([f"P{i}" for i in range(k)], loc="center left", bbox_to_anchor=(1, 0.5), frameon=True)
    fig.tight_layout(pad=0.5)
    return fig, {"chart_family": "pie_exploded", "n_slices": k}


def chart_non_tufte_dashboard(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    fig = plt.figure(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="#ddeeff")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    for ax, c in zip((ax1, ax2, ax3), ["#ffe0e0", "#e0ffe0", "#e0e0ff"]):
        ax.set_facecolor(c)
        _junk_axes(ax, rng)
    x = np.arange(10)
    ax1.bar(x, rng.uniform(0.2, 1, 10), color=_loud_colors(rng, 10))
    ax2.plot(np.linspace(0, 1, 40), np.cumsum(rng.normal(0, 0.05, 40)), linewidth=3, color="magenta")
    ax3.scatter(rng.uniform(0, 1, 80), rng.uniform(0, 1, 80), s=rng.uniform(30, 120, 80), c=rng.random((80, 3)))
    ax1.set_title("Widget A", fontsize=14, color="red")
    ax2.set_title("Widget B", fontsize=14, color="blue")
    ax3.set_title("Scatter Storm", fontsize=14, color="green")
    fig.suptitle("BUSY DASHBOARD", fontsize=18, fontweight="bold", color="#880000")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.1, hspace=0.4, wspace=0.35)
    return fig, {"chart_family": "dashboard", "panels": 3}


def chart_non_tufte_bar3d(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    k = int(rng.integers(5, 10))
    fig = plt.figure(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    xs = np.arange(k)
    ys = np.zeros(k)
    zs = np.zeros(k)
    dx = np.ones(k) * 0.6
    dy = np.ones(k) * 0.6
    dz = rng.uniform(0.3, 1.0, k)
    colors = _loud_colors(rng, k)
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=[c[:3] for c in colors], edgecolor="k", linewidth=0.8)
    ax.set_xticks(xs + 0.3)
    ax.set_xticklabels([f"B{i}" for i in range(k)], rotation=30)
    ax.view_init(elev=float(rng.uniform(20, 40)), azim=float(rng.uniform(40, 80)))
    ax.set_title("3D Bars (chartjunk)", fontsize=13, color="#aa00aa")
    fig.tight_layout(pad=0.4)
    return fig, {"chart_family": "bar3d", "n_bars": k}


def chart_non_tufte_scatter_annotated(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    n = int(rng.integers(25, 70))
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="#f8f8ff")
    ax.scatter(
        x,
        y,
        s=float(rng.uniform(120, 400)),
        c=rng.uniform(0, 1, n),
        cmap="hsv",
        alpha=0.85,
        edgecolors="black",
        linewidths=1.5,
    )
    _junk_axes(ax, rng)
    for _ in range(int(rng.integers(2, 6))):
        i = int(rng.integers(0, n))
        ax.annotate(
            "LOOK!",
            (x[i], y[i]),
            xytext=(x[i] + 0.1, y[i] + 0.1),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            fontsize=10,
            color="darkred",
        )
    ax.legend(["Points"], loc="upper left", fancybox=True, shadow=True)
    fig.tight_layout(pad=0.5)
    return fig, {"chart_family": "scatter_annotated", "n_points": n}


def chart_non_tufte_histogram_busy(
    rng: np.random.Generator, w_px: int, h_px: int, dpi: float
) -> tuple[Figure, dict[str, Any]]:
    n = int(rng.integers(400, 1200))
    data = rng.normal(0, 1, n)
    bins = int(rng.integers(35, 70))
    fig, ax = plt.subplots(figsize=_figsize_inches(w_px, h_px, dpi), dpi=dpi, facecolor="#fff0f0")
    _, _, patches = ax.hist(data, bins=bins, edgecolor="black", linewidth=1.2)
    cmap = plt.get_cmap("gist_rainbow")
    for i, p in enumerate(patches):
        p.set_facecolor(cmap(i / max(len(patches) - 1, 1)))
    _junk_axes(ax, rng)
    ax.set_title("Over-binned histogram", fontsize=15, fontweight="bold")
    ax.legend(["Distribution"], loc="upper right")
    fig.tight_layout(pad=0.5)
    return fig, {"chart_family": "histogram_busy", "n_samples": n, "n_bins": bins}
