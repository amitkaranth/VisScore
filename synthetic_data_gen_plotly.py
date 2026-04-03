"""
VisScore: Synthetic chart dataset using Plotly + Kaleido (non-Matplotlib renderer).

Use alongside synthetic_data_gen.py to diversify visual style (fonts, grids, defaults)
so CNNs rely less on Matplotlib-specific cues.

Output layout matches training expectations:
  <output_dir>/good/*.png
  <output_dir>/bad/*.png
  <output_dir>/metadata_plotly.json

Filenames use prefix "plotly_" by default so you can merge into the same folders as
Matplotlib exports without overwriting.

Requires: pip install plotly kaleido
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit(
        "Install plotly: pip install plotly kaleido\n"
        "Kaleido is required for static PNG export."
    ) from e

AUGMENT_STRENGTH = "medium"


def _ensure_matching_lengths(data, cats):
    data = list(data)
    cats = list(cats)
    n = min(len(data), len(cats))
    if n == 0:
        return [], []
    if len(data) != len(cats):
        data = data[:n]
        cats = cats[:n]
    return data, cats


def random_data(n=6, low=10, high=100):
    return np.random.randint(low, high, size=n).tolist()


def random_categories(n=6):
    pools = [
        ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
        ["Product A", "Product B", "Product C", "Product D", "Product E", "Product F"],
        ["Region 1", "Region 2", "Region 3", "Region 4", "Region 5", "Region 6"],
        ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"],
    ]
    pool = random.choice(pools)
    if n <= len(pool):
        return pool[:n]
    cats = list(pool)
    while len(cats) < n:
        cats.append(f"Cat{len(cats)+1}")
    return cats


def random_title():
    titles = [
        "Sales by Quarter", "Revenue Growth", "Monthly Performance",
        "Regional Comparison", "Annual Metrics", "Team Productivity",
        "Customer Satisfaction", "Market Share Analysis",
    ]
    return random.choice(titles)


def random_time_series(n=12):
    base = datetime(2023, 1, 1)
    dates = [base + timedelta(days=30 * i) for i in range(n)]
    vals = np.cumsum(np.random.randn(n) * 10 + 5) + 50
    return dates, vals.tolist()


def pick_palette_good(n):
    bases = [
        ["#4C72B0"],
        ["#4C72B0", "#DD8452"],
        ["#4C72B0", "#55A868", "#C44E52"],
    ]
    base = random.choice(bases)
    if len(base) >= n:
        return base[:n]
    return (base * ((n // len(base)) + 1))[:n]


def rainbow_colors(n):
    return [f"hsl({int(360 * i / max(n, 1))}, 85%, 55%)" for i in range(n)]


def _layout_base(title, width=880, height=540):
    return dict(
        title=dict(text=title, font=dict(size=18)),
        width=width,
        height=height,
        margin=dict(l=64, r=48, t=72, b=64),
        paper_bgcolor="white",
        font=dict(size=12),
    )


def _update_layout(fig, chart_title, **kwargs):
    """Merge base layout with overrides in one dict (avoids duplicate kw errors in Plotly).

    `chart_title` is the string for the figure title. Plotly layout key `title` may appear
    in **kwargs (e.g. custom font color); do not name this parameter `title` or it clashes.
    """
    fig.update_layout({**_layout_base(chart_title), **kwargs})


def augment_png(path: str, dpi: int = 150) -> None:
    try:
        img = Image.open(path).convert("RGB")
        strength = AUGMENT_STRENGTH
        p_map = {"none": 0.0, "low": 0.2, "medium": 0.45, "high": 0.75}
        p = p_map.get(strength, 0.45)

        if random.random() < p * 0.7:
            scale = random.uniform(0.8, 1.3)
            new_size = (max(32, int(img.width * scale)), max(32, int(img.height * scale)))
            img = img.resize(new_size, resample=Image.BILINEAR)

        if random.random() < p * 0.5:
            angle = random.uniform(-8, 8)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

        if random.random() < p * 0.6:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        if random.random() < p * 0.6:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.2))

        if random.random() < p * 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.6)))

        if random.random() < p * 0.8:
            arr = np.array(img).astype(np.int16)
            noise = np.random.normal(0, random.uniform(2, max(6, p * 20)), arr.shape).astype(
                np.int16
            )
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        img.save(path, dpi=(dpi, dpi))
    except Exception:
        pass


def write_figure(fig: go.Figure, path: str, dpi: int = 150) -> None:
    scale = 2 if dpi >= 150 else 1
    fig.write_image(path, scale=scale)
    augment_png(path, dpi=dpi)


def good_bar_plotly(data, cats, title):
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return go.Figure()
    colors = pick_palette_good(len(data))
    fig = go.Figure(
        data=[
            go.Bar(
                x=cats,
                y=data,
                marker=dict(color=colors, line=dict(width=0)),
                text=[str(v) for v in data],
                textposition="outside",
            )
        ]
    )
    ymax = max(data) * 1.18
    _update_layout(
        fig,
        title,
        yaxis=dict(title="Value", rangemode="tozero", range=[0, ymax], gridcolor="#dddddd"),
        xaxis=dict(title="", showgrid=False),
    )
    return fig


def good_line_plotly(dates, vals, title):
    fig = go.Figure(
        data=[
            go.Scatter(
                x=dates,
                y=vals,
                mode="lines+markers",
                line=dict(color="#4C72B0", width=2),
                marker=dict(size=6),
            )
        ]
    )
    _update_layout(
        fig,
        title,
        yaxis=dict(title="Value", gridcolor="#dddddd"),
        xaxis=dict(title="", showgrid=False),
    )
    return fig


def good_scatter_plotly(title):
    n = random.randint(35, 85)
    x = np.random.randn(n) * 15 + 50
    y = x * random.uniform(0.5, 1.5) + np.random.randn(n) * 10
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(color="#4C72B0", size=9, line=dict(width=0.5, color="white")),
            )
        ]
    )
    _update_layout(
        fig,
        title,
        xaxis=dict(title="Variable X", showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(title="Variable Y", showgrid=True, gridcolor="#eeeeee"),
    )
    return fig


def good_hbar_plotly(data, cats, title):
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return go.Figure()
    colors = pick_palette_good(len(data))
    fig = go.Figure(
        data=[
            go.Bar(
                x=data,
                y=cats,
                orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[str(v) for v in data],
                textposition="outside",
            )
        ]
    )
    _update_layout(
        fig,
        title,
        xaxis=dict(title="Value", rangemode="tozero", gridcolor="#dddddd"),
        yaxis=dict(title="", showgrid=False),
    )
    return fig


def good_pie_plotly(title):
    n = random.choice([2, 3])
    data = random_data(n, 20, 60)
    cats = random_categories(n)
    colors = pick_palette_good(n)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=cats,
                values=data,
                marker=dict(colors=colors, line=dict(color="white", width=1)),
                textinfo="percent+label",
                hole=0.0,
            )
        ]
    )
    _update_layout(fig, title, showlegend=False)
    return fig


def bad_rainbow_bar_plotly(data, cats, title):
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return go.Figure(), []
    cols = rainbow_colors(len(data))
    fig = go.Figure(
        data=[
            go.Bar(
                x=cats,
                y=data,
                marker=dict(color=cols, line=dict(color="white", width=2)),
            )
        ]
    )
    _update_layout(
        fig,
        title,
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#16213e",
        font=dict(color="white", size=12),
        title=dict(text=title, font=dict(size=18, color="white")),
        yaxis=dict(title="Value", gridcolor="rgba(255,255,255,0.2)", color="white"),
        xaxis=dict(color="white"),
    )
    return fig, ["rainbow_palette", "dark_theme_abuse", "low_data_ink_ratio"]


def bad_truncated_bar_plotly(data, cats, title):
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return go.Figure(), []
    fig = go.Figure(
        data=[go.Bar(x=cats, y=data, marker_color="#C44E52")]
    )
    lo = min(data) * random.uniform(0.55, 0.92)
    hi = max(data) * 1.02
    _update_layout(
        fig,
        title,
        yaxis=dict(title="Value", range=[lo, hi], showgrid=True, gridcolor="#ccc"),
        xaxis=dict(showgrid=False),
    )
    return fig, ["truncated_axis", "misleading_scale"]


def bad_heavy_grid_plotly(data, cats, title):
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return go.Figure(), []
    fig = go.Figure(
        data=[
            go.Bar(
                x=cats,
                y=data,
                marker=dict(color="#aaaaaa", line=dict(color="black", width=1)),
            )
        ]
    )
    _update_layout(
        fig,
        title,
        plot_bgcolor="#e8e8e8",
        paper_bgcolor="#f0f0f0",
        xaxis=dict(
            showgrid=True,
            gridwidth=2.5,
            gridcolor="#333333",
            zeroline=True,
            zerolinewidth=2,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=2.5,
            gridcolor="#333333",
            zeroline=True,
            zerolinewidth=2,
        ),
    )
    return fig, ["heavy_gridlines", "low_data_ink_ratio", "data_obscured"]


def bad_dual_axis_plotly(title):
    cats = random_categories(8)
    y1 = np.array(random_data(8, 100, 500))
    y2 = np.random.randn(8) * 2 + 10

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=cats, y=y1, name="Revenue", marker_color="#4C72B0", opacity=0.75),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=cats,
            y=y2,
            name="Satisfaction",
            mode="lines+markers",
            line=dict(color="#C44E52", width=3),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Revenue", secondary_y=False, range=[min(y1) * 0.85, max(y1) * 1.15])
    fig.update_yaxes(title_text="Satisfaction", secondary_y=True, range=[min(y2) - 4, max(y2) + 4])
    _update_layout(fig, title, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(title_text="")
    return fig, ["dual_axis_abuse", "misleading_correlation", "scale_manipulation"]


def bad_spaghetti_plotly(title):
    x = np.arange(12)
    n_lines = random.randint(8, 14)
    fig = go.Figure()
    cols = rainbow_colors(n_lines)
    for i in range(n_lines):
        y = np.cumsum(np.random.randn(12) * 5 + 2) + random.randint(20, 80)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"S{i+1}",
                line=dict(color=cols[i], width=1.8),
            )
        )
    _update_layout(
        fig,
        title,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, font=dict(size=8)),
        yaxis=dict(gridcolor="#ddd"),
        xaxis=dict(title="Month", showgrid=False),
    )
    return fig, ["spaghetti_chart", "too_many_series", "rainbow_palette", "low_readability"]


def bad_3d_bar_plotly(data, cats, title):
    """3D-style chart using Surface (Plotly has no go.Bar3d)."""
    data, cats = _ensure_matching_lengths(data, cats)
    if not len(data):
        return go.Figure(), []
    n = len(data)
    xs = list(range(n))
    x = np.arange(n, dtype=float)
    y = np.array([0.0])
    z = np.array([list(map(float, data))], dtype=float)
    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )
    _update_layout(
        fig,
        title,
        scene=dict(
            xaxis=dict(title="", tickvals=xs, ticktext=cats, tickangle=-25),
            yaxis=dict(visible=False),
            zaxis=dict(title="Value"),
            bgcolor="#fafafa",
        ),
    )
    return fig, ["3d_effects", "perspective_distortion", "chartjunk"]


def bad_chartjunk_bar_plotly(data, cats, title):
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return go.Figure(), []
    fig = go.Figure(
        data=[
            go.Bar(
                x=cats,
                y=data,
                marker=dict(color="#4C72B0", line=dict(color="gold", width=3)),
            )
        ]
    )
    shapes = []
    for _ in range(random.randint(4, 9)):
        shapes.append(
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=random.random() * 0.85,
                y0=random.random() * 0.85,
                x1=random.random() * 0.15 + 0.85,
                y1=random.random() * 0.15 + 0.85,
                fillcolor=random.choice(["#ffeecc", "#ddeeff", "#ffddee"]),
                opacity=0.55,
                line_width=0,
            )
        )
    _update_layout(
        fig,
        title,
        shapes=shapes,
        annotations=[
            dict(
                text="★ INSIGHT ★",
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                showarrow=False,
                font=dict(size=22, color="crimson"),
            )
        ],
        yaxis=dict(gridcolor="#bbbbbb", gridwidth=1.5),
    )
    return fig, ["chartjunk", "decorative_elements", "low_data_ink_ratio"]


def generate_sample(idx: int, output_dir: str, prefix: str):
    n_cats = random.randint(2, 10)
    low = random.randint(1, 20)
    high = random.randint(low + 8, low + 220)
    data = random_data(n_cats, low=low, high=high)
    cats = random_categories(n_cats)
    title = random_title()
    dates, time_vals = random_time_series()

    good_type = random.choice(["bar", "line", "scatter", "hbar", "pie"])
    if good_type == "bar":
        fig_g = good_bar_plotly(data, cats, title)
    elif good_type == "line":
        fig_g = good_line_plotly(dates, time_vals, title)
    elif good_type == "scatter":
        fig_g = good_scatter_plotly(title)
    elif good_type == "hbar":
        fig_g = good_hbar_plotly(data, cats, title)
    else:
        fig_g = good_pie_plotly(title)

    gid = f"{prefix}good_{idx:05d}"
    good_path = os.path.join(output_dir, "good", f"{gid}.png")
    write_figure(fig_g, good_path)
    good_meta = {
        "id": gid,
        "label": "good",
        "chart_type": good_type,
        "title": title,
        "violations": [],
        "tufte_score": 1.0,
        "renderer": "plotly_kaleido",
    }

    bad_choice = random.choice(
        [
            "rainbow",
            "truncated",
            "heavy_grid",
            "dual_axis",
            "spaghetti",
            "3d",
            "chartjunk",
        ]
    )
    if bad_choice == "rainbow":
        fig_b, violations = bad_rainbow_bar_plotly(data, cats, title)
    elif bad_choice == "truncated":
        fig_b, violations = bad_truncated_bar_plotly(data, cats, title)
    elif bad_choice == "heavy_grid":
        fig_b, violations = bad_heavy_grid_plotly(data, cats, title)
    elif bad_choice == "dual_axis":
        fig_b, violations = bad_dual_axis_plotly(title)
    elif bad_choice == "spaghetti":
        fig_b, violations = bad_spaghetti_plotly(title)
    elif bad_choice == "3d":
        fig_b, violations = bad_3d_bar_plotly(data, cats, title)
    else:
        fig_b, violations = bad_chartjunk_bar_plotly(data, cats, title)

    bid = f"{prefix}bad_{idx:05d}"
    bad_path = os.path.join(output_dir, "bad", f"{bid}.png")
    write_figure(fig_b, bad_path)
    bad_meta = {
        "id": bid,
        "label": "bad",
        "chart_type": bad_choice,
        "title": title,
        "violations": violations,
        "tufte_score": 0.0,
        "renderer": "plotly_kaleido",
    }

    return good_meta, bad_meta


def main():
    global AUGMENT_STRENGTH
    parser = argparse.ArgumentParser(
        description="Generate Plotly/Kaleido synthetic charts for VisScore (style diversity)."
    )
    parser.add_argument("--output_dir", type=str, default="./vis_dataset_plotly",
                        help="Output root (good/, bad/, metadata file)")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of good/bad pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment_strength",
        choices=["none", "low", "medium", "high"],
        default="medium",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="plotly_",
        help="Filename prefix to avoid collisions when merging with Matplotlib dataset",
    )
    parser.add_argument("--seed_aug", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    AUGMENT_STRENGTH = args.augment_strength
    if args.seed_aug is not None:
        random.seed(args.seed_aug)
        np.random.seed(args.seed_aug)

    os.makedirs(os.path.join(args.output_dir, "good"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "bad"), exist_ok=True)

    prefix = args.prefix.strip() or "plotly_"
    if not prefix.endswith("_"):
        prefix = prefix + "_"

    all_meta = []
    print(f"[plotly] Generating {args.num_samples} good/bad pairs → {args.output_dir}")

    for i in range(args.num_samples):
        gm, bm = generate_sample(i, args.output_dir, prefix=prefix)
        all_meta.extend([gm, bm])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.num_samples} pairs")

    meta_path = os.path.join(args.output_dir, "metadata_plotly.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2, default=str)

    violation_counts = {}
    for m in all_meta:
        for v in m["violations"]:
            violation_counts[v] = violation_counts.get(v, 0) + 1

    print(f"\nDone. Total images: {len(all_meta)}")
    print(f"  Metadata: {meta_path}")
    print("  Violation counts:")
    for v, c in sorted(violation_counts.items(), key=lambda x: -x[1]):
        print(f"    {v}: {c}")
    print(
        "\nMerge tip: copy good/*.png and bad/*.png into your main vis_dataset folders, "
        "then train with --data_dir pointing at the combined dataset."
    )


if __name__ == "__main__":
    main()
