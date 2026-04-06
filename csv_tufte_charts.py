#!/usr/bin/env python3
"""
Standalone utility: read one or more CSV files and export chart images (PNG).

For each inferred chart type, writes a **good** (Tufte-aligned) version and, by default,
a **bad** version (chartjunk, heavy grids, rainbow abuse, etc.) suitable for contrast / training.

This script is independent of VisScore training, inference, Streamlit, and synthetic_data_gen.
It only uses numpy + matplotlib (already in the project environment).

Good charts: high data-ink, white background, light horizontal grid only, honest bar baseline,
single-hue categorical bars, numeric columns chosen to favor real metrics over IDs/ISO codes.
Bad charts: low data-ink, tinted backgrounds, dense grids, decorative styling, rainbow palettes,
optional misleading bar baseline (truncated).

Usage:
  python csv_tufte_charts.py --input_dir /path/to/csv_folder --output_dir ./tufte_from_csv
  python csv_tufte_charts.py --input_glob "./data/*.csv" --output_dir ./out --no_bad

`--max_charts_per_file` caps how many *chart types* are emitted per CSV; each type may produce
both good and bad PNGs unless `--no_bad` is set.

Dependencies: numpy and matplotlib (same as the main VisScore `requirements.txt`; use your project venv).
"""

from __future__ import annotations

import argparse
import csv
import glob as glob_module
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Tufte-friendly palette (colorblind-safe, not rainbow)
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
# Good categorical bars: single hue (matches synthetic "good" training style; avoids rainbow false BAD)
_GOOD_BAR_COLOR = "#4C72B0"

_BAD_BAR_FILL = ["#ff6b6b", "#feca57", "#48dbfb", "#ff9ff3", "#54a0ff", "#5f27cd", "#1dd1a1", "#ff9f43"]
_BAD_HATCHES = ["///", r"\\\\", "xxx", "+++", "ooo", "..."]


def _apply_tufte_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.45, color="#cccccc")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[-\s]+", "_", s.strip())
    return s[:80] or "col"


def _try_float(x: str) -> float | None:
    if x is None:
        return None
    t = str(x).strip()
    if not t or t.lower() in ("nan", "null", "none", "n/a", "na", "-"):
        return None
    t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def _try_datetime(x: str) -> datetime | None:
    t = str(x).strip()
    if not t:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d-%b-%Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(t[:26], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00").split(".")[0])
    except ValueError:
        return None


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with path.open(newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return [], []
                fieldnames = [h.strip() if h else f"col{i}" for i, h in enumerate(reader.fieldnames)]
                rows: list[dict[str, str]] = []
                for raw in reader:
                    row = {fieldnames[i]: (raw.get(reader.fieldnames[i], "") or "").strip() for i in range(len(fieldnames))}
                    rows.append(row)
                return fieldnames, rows
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return [], []


def _infer_columns(fieldnames: list[str], rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    n = max(len(rows), 1)
    info: dict[str, dict[str, Any]] = {}
    for name in fieldnames:
        vals = [rows[i].get(name, "") for i in range(len(rows))]
        float_ok = sum(1 for v in vals if _try_float(v) is not None)
        dt_ok = sum(1 for v in vals if _try_datetime(v) is not None)
        nonempty = sum(1 for v in vals if str(v).strip() != "")
        info[name] = {
            "numeric_ratio": float_ok / max(nonempty, 1),
            "datetime_ratio": dt_ok / max(nonempty, 1),
            "nonempty": nonempty,
        }
    return info


def _table_from_csv(path: Path) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, Any]]]:
    fieldnames, rows = _read_csv_rows(path)
    if not fieldnames or not rows:
        return fieldnames, rows, {}
    meta = _infer_columns(fieldnames, rows)
    return fieldnames, rows, meta


def _norm_header(name: str) -> str:
    return re.sub(r"[_\s]+", " ", name.strip().lower())


def _numeric_axis_priority(col_name: str) -> int:
    """Score for using a column as the *value* axis (bar length, etc.). Higher = prefer over IDs/codes."""
    n = _norm_header(col_name)
    s = 0
    metric_terms = (
        "burden",
        "death",
        "deaths",
        "mortality",
        "incidence",
        "prevalence",
        "case",
        "cases",
        "notification",
        "notified",
        "rate",
        "rates",
        "percent",
        "percentage",
        "proportion",
        "share",
        "total",
        "count",
        "sum",
        "mean",
        "average",
        "avg",
        "median",
        "population",
        "estimate",
        "estimated",
        "value",
        "amount",
        "cost",
        "revenue",
        "spend",
        "budget",
        "daly",
        "yll",
        "yld",
        "life expectancy",
        "coverage",
        "risk",
        "hazard",
        "odds",
        "ratio",
        "index",
        "score",
        "weight",
        "height",
        "mass",
        "volume",
        "temperature",
        "pressure",
        "growth",
        "change",
        "difference",
    )
    for term in metric_terms:
        if term in n:
            s += 12
    disease_markers = ("tb ", " tb", "tuberc", "hiv", "malaria", "covid", "measles", "mortality")
    for term in disease_markers:
        if term in n:
            s += 8
    if re.search(r"\b(100k|100\s*k|per\s*100|/100)\b", n) or re.search(
        r"\b(inc|prev|mort|notif|mdr|tbhiv|smear|detection)\b", n
    ):
        s += 14
    if re.search(r"\biso\b", n):
        s -= 80
    if "country" in n and "code" in n:
        s -= 70
    if "territory" in n and "code" in n:
        s -= 70
    if "numeric" in n and "code" in n and ("country" in n or "territory" in n):
        s -= 75
    if re.search(r"\b(code|no\.?|number)\b", n) and (
        "country" in n or "territory" in n or "iso" in n or "fips" in n or "region" in n
    ):
        s -= 60
    if re.search(r"\b(postal|zipcode|zip code|phone|fax|latitude|longitude|geocode)\b", n):
        s -= 55
    if re.search(r"\b(row|record)\s*id\b|_id$|^id$|\bid uuid\b", n):
        s -= 45
    if re.search(r"\bordinal\b|\brank\b|\bsequence\b", n) and "death" not in n:
        s -= 15
    return s


def _pick_columns(
    fieldnames: list[str], meta: dict[str, dict[str, Any]]
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (date_col, cat_col, num_col_primary, num_col_secondary) hints."""
    date_candidates = [c for c in fieldnames if meta[c]["datetime_ratio"] >= 0.7 and meta[c]["nonempty"] > 0]
    num_candidates = [c for c in fieldnames if meta[c]["numeric_ratio"] >= 0.85 and meta[c]["nonempty"] > 0]
    cat_candidates = [
        c
        for c in fieldnames
        if c not in date_candidates
        and meta[c]["numeric_ratio"] < 0.5
        and meta[c]["nonempty"] > 0
    ]
    date_col = date_candidates[0] if date_candidates else None
    nums_raw = [c for c in num_candidates if c != date_col]
    nums = sorted(nums_raw, key=lambda c: (-_numeric_axis_priority(c), fieldnames.index(c)))
    num_a = nums[0] if nums else None
    num_b = nums[1] if len(nums) > 1 else None
    cat_col = None
    for c in cat_candidates:
        if c not in (date_col, num_a, num_b):
            cat_col = c
            break
    if cat_col is None and nums:
        for c in fieldnames:
            if c not in nums and c != date_col and meta[c]["nonempty"] > 0:
                cat_col = c
                break
    return date_col, cat_col, num_a, num_b


def _save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def chart_line(dates: list[datetime], y: list[float], ylabel: str, title: str, dpi: int, out_path: Path) -> None:
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    xd = [dates[i] for i in order]
    yv = [y[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.plot(xd, yv, color=_PALETTE[0], linewidth=1.8, clip_on=False)
    _apply_tufte_axes(ax)
    ax.set_title(title, fontsize=12, fontweight="normal", pad=10, color="#111111")
    ax.set_ylabel(ylabel, fontsize=10, color="#333333")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def chart_bar(categories: list[str], values: list[float], ylabel: str, title: str, dpi: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, min(10, 0.35 * len(categories)))), facecolor="white")
    y_pos = np.arange(len(categories))
    ax.barh(y_pos, values, color=_GOOD_BAR_COLOR, edgecolor="none", height=0.65)
    ax.set_yticks(y_pos, labels=categories, fontsize=9)
    ax.invert_yaxis()
    _apply_tufte_axes(ax)
    ax.set_title(title, fontsize=12, fontweight="normal", pad=10)
    ax.set_xlabel(ylabel, fontsize=10)
    xmax = max(values) if values else 1.0
    ax.set_xlim(0, xmax * 1.08)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def chart_bar_vertical(categories: list[str], values: list[float], ylabel: str, title: str, dpi: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(min(10, max(6, 0.45 * len(categories))), 5), facecolor="white")
    x_pos = np.arange(len(categories))
    ax.bar(x_pos, values, color=_GOOD_BAR_COLOR, edgecolor="none", width=0.65)
    ax.set_xticks(x_pos, labels=categories, rotation=45, ha="right", fontsize=9)
    _apply_tufte_axes(ax)
    ax.set_title(title, fontsize=12, fontweight="normal", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ymax = max(values) if values else 1.0
    ax.set_ylim(0, ymax * 1.12)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def chart_scatter(x: list[float], y: list[float], xlabel: str, ylabel: str, title: str, dpi: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="white")
    ax.scatter(x, y, s=28, color=_PALETTE[0], edgecolors="none", alpha=0.85)
    _apply_tufte_axes(ax)
    ax.set_title(title, fontsize=12, fontweight="normal", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def chart_hist(values: list[float], title: str, xlabel: str, dpi: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    n = len(values)
    bins = max(10, min(40, int(round(math.sqrt(n)))))
    ax.hist(values, bins=bins, color=_PALETTE[0], edgecolor="white", linewidth=0.6)
    _apply_tufte_axes(ax)
    ax.set_title(title, fontsize=12, fontweight="normal", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


# --- "Bad" charts (Tufte violations; mirrors good chart types) ---


def chart_line_bad(dates: list[datetime], y: list[float], ylabel: str, title: str, dpi: int, out_path: Path) -> None:
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    xd = [dates[i] for i in order]
    yv = [y[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#e8e0d4")
    ax.set_facecolor("#c5e3f6")
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_edgecolor("#8B0000")
    ax.tick_params(axis="both", which="both", length=6, width=2, labelsize=11, colors="#4a004a")
    ax.grid(True, which="major", linewidth=1.1, alpha=0.85, color="#444444")
    ax.grid(True, which="minor", linewidth=0.6, alpha=0.5, color="#888888")
    ax.minorticks_on()
    ax.set_axisbelow(False)
    nseg = max(1, len(xd) - 1)
    for i in range(len(xd) - 1):
        c = plt.cm.hsv(i / max(1, nseg - 1))
        ax.plot(xd[i : i + 2], yv[i : i + 2], color=c, linewidth=4.0, solid_capstyle="round")
    ax.set_title(
        title.upper(),
        fontsize=15,
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="yellow", edgecolor="red", linewidth=2),
    )
    ax.set_ylabel(ylabel + " !!!", fontsize=12, fontweight="bold", color="navy")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def chart_bar_bad(categories: list[str], values: list[float], ylabel: str, title: str, dpi: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, min(10, 0.35 * len(categories)))), facecolor="#f0e8dc")
    ax.set_facecolor("#e8f4fc")
    y_pos = np.arange(len(categories))
    for i in range(len(categories)):
        ax.barh(
            y_pos[i],
            values[i],
            color=_BAD_BAR_FILL[i % len(_BAD_BAR_FILL)],
            edgecolor="black",
            linewidth=2,
            hatch=_BAD_HATCHES[i % len(_BAD_HATCHES)],
            height=0.72,
        )
    ax.set_yticks(y_pos, labels=categories, fontsize=10)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_edgecolor("black")
    ax.tick_params(axis="both", which="both", length=8, width=2)
    ax.grid(True, axis="both", linewidth=1.5, alpha=0.75, color="gray")
    ax.set_axisbelow(False)
    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        fontfamily="serif",
        color="darkviolet",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", edgecolor="purple", linewidth=2),
    )
    ax.set_xlabel(ylabel, fontsize=12, fontweight="bold", color="darkgreen")
    xmax = max(values) if values else 1.0
    floor = xmax * 0.55
    ax.set_xlim(floor, xmax * 1.12)
    patches = [mpatches.Patch(color=_BAD_BAR_FILL[i % len(_BAD_BAR_FILL)], label=categories[i]) for i in range(len(categories))]
    ax.legend(
        handles=patches,
        loc="lower right",
        fontsize=8,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def chart_bar_vertical_bad(
    categories: list[str], values: list[float], ylabel: str, title: str, dpi: int, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(min(10, max(6, 0.45 * len(categories))), 5), facecolor="#f5f0e8")
    ax.set_facecolor("#ddeeff")
    x_pos = np.arange(len(categories))
    for i in range(len(categories)):
        ax.bar(
            x_pos[i],
            values[i],
            color=_BAD_BAR_FILL[i % len(_BAD_BAR_FILL)],
            edgecolor="black",
            linewidth=2,
            hatch=_BAD_HATCHES[i % len(_BAD_HATCHES)],
            width=0.68,
        )
    ax.set_xticks(x_pos, labels=categories, rotation=55, ha="right", fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_edgecolor("#003366")
    ax.tick_params(axis="both", which="both", length=8, width=2)
    ax.grid(True, axis="both", linewidth=1.8, alpha=0.8, color="#555555")
    ax.set_axisbelow(False)
    ax.set_title(title, fontsize=17, fontweight="bold", color="crimson", style="italic")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color="darkorange")
    ymax = max(values) if values else 1.0
    ymin = ymax * 0.5
    ax.set_ylim(ymin, ymax * 1.25)
    patches = [mpatches.Patch(color=_BAD_BAR_FILL[i % len(_BAD_BAR_FILL)], label=categories[i]) for i in range(len(categories))]
    ax.legend(handles=patches, loc="upper left", fontsize=8, fancybox=True, shadow=True)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def chart_scatter_bad(x: list[float], y: list[float], xlabel: str, ylabel: str, title: str, dpi: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="#eae2d6")
    ax.set_facecolor("#d4ecf8")
    xa = np.array(x, dtype=float)
    c = plt.cm.rainbow((xa - xa.min()) / max(1e-9, xa.max() - xa.min()))
    ax.scatter(x, y, s=140, c=c, edgecolors="black", linewidths=1.2, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("darkred")
    ax.tick_params(axis="both", which="both", length=7, width=2)
    ax.grid(True, axis="both", linewidth=1.2, alpha=0.85, color="dimgray", linestyle="--")
    ax.set_axisbelow(False)
    ax.set_title(title, fontsize=15, fontweight="bold", color="purple")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def chart_hist_bad(values: list[float], title: str, xlabel: str, dpi: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#ede5d8")
    ax.set_facecolor("#cfe8ff")
    n = len(values)
    bins = max(10, min(40, int(round(math.sqrt(n)))))
    n_art, bin_edges, patches_art = ax.hist(values, bins=bins, edgecolor="black", linewidth=1.2)
    for i, p in enumerate(patches_art):
        p.set_facecolor(plt.cm.jet(i / max(1, len(patches_art) - 1)))
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    ax.tick_params(axis="both", which="both", length=6, width=2)
    ax.grid(True, axis="both", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(False)
    ax.set_title(title + " — WOW", fontsize=14, fontweight="bold", color="darkred")
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel("COUNT", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def _aggregate_bar(rows: list[dict[str, str]], cat_col: str, val_col: str) -> tuple[list[str], list[float]]:
    sums: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for r in rows:
        k = str(r.get(cat_col, "")).strip() or "(blank)"
        v = _try_float(r.get(val_col, ""))
        if v is None:
            continue
        sums[k] += v
        counts[k] += 1
    items = list(sums.items())
    items.sort(key=lambda t: t[1], reverse=True)
    max_cats = 24
    if len(items) > max_cats:
        items = items[:max_cats]
    cats = [t[0] for t in items]
    vals = [t[1] for t in items]
    return cats, vals


def _line_series(rows: list[dict[str, str]], date_col: str, val_col: str) -> tuple[list[datetime], list[float]] | None:
    pairs: list[tuple[datetime, float]] = []
    for r in rows:
        d = _try_datetime(r.get(date_col, ""))
        v = _try_float(r.get(val_col, ""))
        if d is None or v is None:
            continue
        pairs.append((d, v))
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda t: t[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _scatter_series(rows: list[dict[str, str]], xc: str, yc: str) -> tuple[list[float], list[float]] | None:
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        x = _try_float(r.get(xc, ""))
        y = _try_float(r.get(yc, ""))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        return None
    return xs, ys


def _hist_series(rows: list[dict[str, str]], val_col: str) -> list[float] | None:
    vals = []
    for r in rows:
        v = _try_float(r.get(val_col, ""))
        if v is not None:
            vals.append(v)
    return vals if len(vals) >= 3 else None


_LINE_BAD_VIOLATIONS = [
    "heavy_gridlines",
    "low_data_ink_ratio",
    "rainbow_palette",
    "chartjunk",
    "decorative_title",
]
_BAR_BAD_VIOLATIONS = [
    "chartjunk",
    "low_data_ink_ratio",
    "heavy_gridlines",
    "truncated_axis",
    "rainbow_palette",
    "excessive_legend",
]
_SCATTER_BAD_VIOLATIONS = [
    "rainbow_palette",
    "heavy_gridlines",
    "low_data_ink_ratio",
    "chartjunk",
]
_HIST_BAD_VIOLATIONS = [
    "rainbow_palette",
    "jet_colormap",
    "heavy_gridlines",
    "low_data_ink_ratio",
    "chartjunk",
]


def process_csv(
    path: Path,
    output_dir: Path,
    dpi: int,
    max_charts: int,
    stem_prefix: str,
    *,
    include_bad: bool,
) -> list[dict[str, Any]]:
    fieldnames, rows, meta = _table_from_csv(path)
    manifest_rows: list[dict[str, Any]] = []
    if not rows or not fieldnames:
        manifest_rows.append({"source_csv": str(path), "status": "skipped_empty", "charts": []})
        return manifest_rows

    date_col, cat_col, num_a, num_b = _pick_columns(fieldnames, meta)
    base = output_dir / stem_prefix
    base.mkdir(parents=True, exist_ok=True)
    charts: list[dict[str, Any]] = []
    n_out = 0
    line_used_value_col: str | None = None

    def record(kind: str, quality: str, filename: str, rel: str, violations: list[str]) -> None:
        charts.append({"kind": kind, "quality": quality, "file": rel, "violations": violations})

    def use_slot() -> None:
        nonlocal n_out
        n_out += 1

    title_base = path.stem.replace("_", " ")

    if date_col and num_a and n_out < max_charts:
        line_data = _line_series(rows, date_col, num_a)
        if line_data:
            ds, ys = line_data
            fn = f"{stem_prefix}_line_{_slug(date_col)}_{_slug(num_a)}.png"
            chart_line(ds, ys, num_a, title_base, dpi, base / fn)
            record("line", "good", fn, str(Path(stem_prefix) / fn), [])
            if include_bad:
                fnb = f"{stem_prefix}_bad_line_{_slug(date_col)}_{_slug(num_a)}.png"
                chart_line_bad(ds, ys, num_a, title_base, dpi, base / fnb)
                record("line", "bad", fnb, str(Path(stem_prefix) / fnb), list(_LINE_BAD_VIOLATIONS))
            line_used_value_col = num_a
            use_slot()

    if cat_col and num_a and n_out < max_charts:
        cats, vals = _aggregate_bar(rows, cat_col, num_a)
        if len(cats) >= 2:
            use_horizontal = len(max(cats, key=len)) > 14 or len(cats) > 10
            fn = f"{stem_prefix}_bar_{_slug(cat_col)}_{_slug(num_a)}.png"
            if use_horizontal:
                chart_bar(cats, vals, num_a, title_base, dpi, base / fn)
                if include_bad:
                    fnb = f"{stem_prefix}_bad_bar_{_slug(cat_col)}_{_slug(num_a)}.png"
                    chart_bar_bad(cats, vals, num_a, title_base, dpi, base / fnb)
            else:
                chart_bar_vertical(cats, vals, num_a, title_base, dpi, base / fn)
                if include_bad:
                    fnb = f"{stem_prefix}_bad_bar_{_slug(cat_col)}_{_slug(num_a)}.png"
                    chart_bar_vertical_bad(cats, vals, num_a, title_base, dpi, base / fnb)
            record("bar", "good", fn, str(Path(stem_prefix) / fn), [])
            if include_bad:
                record("bar", "bad", fnb, str(Path(stem_prefix) / fnb), list(_BAR_BAD_VIOLATIONS))
            use_slot()

    if num_a and num_b and n_out < max_charts:
        sc = _scatter_series(rows, num_a, num_b)
        if sc:
            xs, ys = sc
            fn = f"{stem_prefix}_scatter_{_slug(num_a)}_{_slug(num_b)}.png"
            chart_scatter(xs, ys, num_a, num_b, title_base, dpi, base / fn)
            record("scatter", "good", fn, str(Path(stem_prefix) / fn), [])
            if include_bad:
                fnb = f"{stem_prefix}_bad_scatter_{_slug(num_a)}_{_slug(num_b)}.png"
                chart_scatter_bad(xs, ys, num_a, num_b, title_base, dpi, base / fnb)
                record("scatter", "bad", fnb, str(Path(stem_prefix) / fnb), list(_SCATTER_BAD_VIOLATIONS))
            use_slot()

    if num_a and n_out < max_charts and line_used_value_col != num_a:
        hv = _hist_series(rows, num_a)
        if hv:
            fn = f"{stem_prefix}_hist_{_slug(num_a)}.png"
            chart_hist(hv, title_base, num_a, dpi, base / fn)
            record("histogram", "good", fn, str(Path(stem_prefix) / fn), [])
            if include_bad:
                fnb = f"{stem_prefix}_bad_hist_{_slug(num_a)}.png"
                chart_hist_bad(hv, title_base, num_a, dpi, base / fnb)
                record("histogram", "bad", fnb, str(Path(stem_prefix) / fnb), list(_HIST_BAD_VIOLATIONS))
            use_slot()

    manifest_rows.append(
        {
            "source_csv": str(path.resolve()),
            "status": "ok" if charts else "no_charts_inferred",
            "inferred": {"date_col": date_col, "category_col": cat_col, "numeric_cols": [c for c in (num_a, num_b) if c]},
            "charts": charts,
        }
    )
    return manifest_rows


def _collect_csv_paths(input_dir: Path | None, input_glob: str | None) -> list[Path]:
    paths: list[Path] = []
    if input_glob:
        for raw in glob_module.glob(input_glob, recursive=False):
            p = Path(raw)
            if p.is_file():
                paths.append(p)
    if input_dir:
        p = Path(input_dir)
        if p.is_dir():
            paths.extend(p.glob("*.csv"))
            paths.extend(p.glob("*.CSV"))
    uniq: dict[str, Path] = {}
    for p in paths:
        if p.is_file():
            uniq[str(p.resolve())] = p
    return sorted(uniq.values(), key=lambda x: str(x).lower())


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Tufte-style charts from CSV files (standalone).")
    ap.add_argument("--input_dir", type=str, default=None, help="Directory containing .csv files")
    ap.add_argument("--input_glob", type=str, default=None, help='Glob such as "./data/*.csv"')
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for PNGs and manifest")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--max_charts_per_file",
        type=int,
        default=6,
        help="Max chart *types* per CSV (line, bar, …). Each type can emit good + bad PNGs.",
    )
    ap.add_argument(
        "--no_bad",
        action="store_true",
        help="Only write Tufte-style (good) charts; skip low data-ink / chartjunk variants.",
    )
    args = ap.parse_args()

    if not args.input_dir and not args.input_glob:
        ap.error("Provide --input_dir and/or --input_glob")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_paths = _collect_csv_paths(Path(args.input_dir) if args.input_dir else None, args.input_glob)
    if not csv_paths:
        print("No CSV files found. Check --input_dir or --input_glob.")
        return

    manifest: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        stem_prefix = _slug(csv_path.stem)
        try:
            manifest.extend(
                process_csv(
                    csv_path,
                    out_root,
                    args.dpi,
                    args.max_charts_per_file,
                    stem_prefix,
                    include_bad=not args.no_bad,
                )
            )
        except Exception as e:
            manifest.append({"source_csv": str(csv_path.resolve()), "status": "error", "error": str(e), "charts": []})

    manifest_path = out_root / "csv_tufte_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(csv_paths)} CSV(s); manifest: {manifest_path}")


if __name__ == "__main__":
    main()
