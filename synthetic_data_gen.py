"""
VisScore: Synthetic Visualization Dataset Generator
Generates pairs of good/bad chart images based on Tufte's principles.

Each chart is saved with metadata (JSON) describing which violations are present.
Good charts follow Tufte's principles; bad charts introduce specific violations.

Violations modeled:
  1. Chartjunk (excessive gridlines, backgrounds, borders)
  2. 3D effects (perspective distortion)
  3. Truncated/misleading axes (non-zero baseline)
  4. Low data-ink ratio (heavy decoration, redundant elements)
  5. Inappropriate chart type (pie chart with too many categories)
  6. Rainbow/poor color palette
  7. Excessive labels/annotations
  8. Dual axes abuse

Usage:
  python synthetic_data_gen.py --output_dir ./dataset --num_samples 500
"""

import os
import json
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from datetime import datetime, timedelta


def _ensure_matching_lengths(data, cats):
    """Ensure `data` and `cats` are lists of the same positive length.

    Trims the longer list to match the shorter. Returns two lists.
    """
    data = list(data)
    cats = list(cats)
    n = min(len(data), len(cats))
    if n == 0:
        return [], []
    if len(data) != len(cats):
        data = data[:n]
        cats = cats[:n]
    return data, cats


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def random_data(n=6, low=10, high=100):
    """Generate random numerical data."""
    return np.random.randint(low, high, size=n).tolist()

def random_categories(n=6):
    """Generate random category labels."""
    pools = [
        ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
        ["Product A", "Product B", "Product C", "Product D", "Product E", "Product F"],
        ["Region 1", "Region 2", "Region 3", "Region 4", "Region 5", "Region 6"],
        ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"],
        ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"],
        ["Dept A", "Dept B", "Dept C", "Dept D", "Dept E", "Dept F"],
    ]
    pool = random.choice(pools)
    return pool[:n]

def random_title():
    titles = [
        "Sales by Quarter", "Revenue Growth", "Monthly Performance",
        "Regional Comparison", "Annual Metrics", "Team Productivity",
        "Customer Satisfaction", "Market Share Analysis",
        "Budget Allocation", "Quarterly Earnings",
        "User Engagement", "Product Adoption Rate",
    ]
    return random.choice(titles)

def random_time_series(n=12):
    """Generate a random time series with dates."""
    base = datetime(2023, 1, 1)
    dates = [base + timedelta(days=30 * i) for i in range(n)]
    vals = np.cumsum(np.random.randn(n) * 10 + 5) + 50
    return dates, vals.tolist()

def save_chart(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


# ============================================================
# GOOD CHART GENERATORS (Tufte-compliant)
# ============================================================

def good_bar_chart(data, cats, title):
    """Clean bar chart: minimal gridlines, high data-ink ratio, zero baseline."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    # Ensure data and category lists match lengths to avoid broadcasting errors
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return fig
    colors = ['#4C72B0'] * len(data)
    ax.bar(cats, data, color=colors, edgecolor='none', width=0.6)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Value", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_ylim(0, max(data) * 1.15)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    for i, v in enumerate(data):
        ax.text(i, v + max(data) * 0.02, str(v), ha='center', va='bottom', fontsize=9, color='#333333')
    fig.tight_layout()
    return fig

def good_line_chart(dates, vals, title):
    """Clean line chart with minimal design."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(dates, vals, color='#4C72B0', linewidth=2, marker='o', markersize=4)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Value", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='both', length=0)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def good_scatter_chart(title):
    """Clean scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    n = random.randint(30, 80)
    x = np.random.randn(n) * 15 + 50
    y = x * random.uniform(0.5, 1.5) + np.random.randn(n) * 10
    ax.scatter(x, y, color='#4C72B0', alpha=0.7, edgecolors='white', linewidth=0.5, s=50)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Variable X", fontsize=10)
    ax.set_ylabel("Variable Y", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    fig.tight_layout()
    return fig

def good_horizontal_bar(data, cats, title):
    """Clean horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    # Ensure data and category lists match lengths to avoid broadcasting errors
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return fig  # nothing to plot

    y_pos = np.arange(len(data))
    ax.barh(y_pos, data, color='#4C72B0', edgecolor='none', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Value", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.grid(True, linewidth=0.3, alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    for i, v in enumerate(data):
        ax.text(v + max(data) * 0.02, i, str(v), ha='left', va='center', fontsize=9, color='#333333')
    fig.tight_layout()
    return fig

def good_pie_chart_simple(title):
    """Pie chart with 3 or fewer categories (acceptable use case)."""
    n = random.choice([2, 3])
    data = random_data(n, 20, 60)
    cats = random_categories(n)
    colors = ['#4C72B0', '#DD8452', '#55A868'][:n]
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    wedges, texts, autotexts = ax.pie(
        data, labels=cats, autopct='%1.1f%%', colors=colors,
        startangle=90, textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color('white')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    return fig


# ============================================================
# BAD CHART GENERATORS (Tufte violations)
# ============================================================

def bad_chartjunk_bar(data, cats, title):
    """Bar chart with heavy chartjunk: background, moiré patterns, borders, 
    excessive gridlines, gradient fills, unnecessary legend."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # Ensure lengths match to avoid broadcasting errors when plotting
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return fig, []
    fig.patch.set_facecolor('#e8e0d4')
    ax.set_facecolor('#f5f0e8')
    hatches = ['///', '\\\\\\', 'xxx', '+++', 'ooo', '...']
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff', '#5f27cd']
    for i, (d, c) in enumerate(zip(data, cats)):
        ax.bar(c, d, color=colors[i % len(colors)], edgecolor='black', linewidth=2,
               hatch=hatches[i % len(hatches)])
    ax.set_title(title, fontsize=16, fontweight='bold',
                 fontfamily='fantasy', color='darkred',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='red', linewidth=2))
    ax.set_ylabel("VALUE!!!", fontsize=14, fontweight='bold', color='red')
    ax.grid(True, linewidth=2, alpha=0.8, color='gray')
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_edgecolor('black')
    ax.tick_params(axis='both', which='both', length=8, width=2, labelsize=12)
    # Unnecessary legend
    patches = [mpatches.Patch(color=colors[i % len(colors)], label=c) for i, c in enumerate(cats)]
    ax.legend(handles=patches, loc='upper left', fontsize=9,
              fancybox=True, shadow=True, borderpad=1.5,
              facecolor='lightyellow', edgecolor='red')
    ax.set_ylim(0, max(data) * 1.3)
    fig.tight_layout()
    return fig, ["chartjunk", "low_data_ink_ratio", "excessive_decoration"]

def bad_3d_bar(data, cats, title):
    """Unnecessary 3D bar chart — perspective distortion makes comparison difficult."""
    fig = plt.figure(figsize=(9, 6))
    # Ensure lengths match
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return fig, ["3d_effect"]
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(len(cats))
    colors = cm.rainbow(np.linspace(0, 1, len(data)))
    ax.bar3d(x, 0, 0, 0.6, 0.6, data, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels(cats, fontsize=8, rotation=15)
    ax.set_zlabel("Value", fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.view_init(elev=25, azim=-45)
    fig.tight_layout()
    return fig, ["3d_effect", "perspective_distortion", "rainbow_palette"]

def bad_truncated_axis(data, cats, title):
    """Bar chart with truncated y-axis (non-zero baseline) — exaggerates differences."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    # ensure matching lengths before plotting
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return fig, []  # nothing to plot

    ax.bar(cats, data, color='#4C72B0', edgecolor='none', width=0.6)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Value", fontsize=10)
    min_val = min(data)
    ax.set_ylim(min_val * 0.85, max(data) * 1.05)  # Non-zero baseline!
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, ["truncated_axis", "misleading_baseline", "graphical_integrity_violation"]

def bad_pie_many_categories(title):
    """Pie chart with too many categories — nearly impossible to read."""
    n = random.randint(10, 18)
    data = random_data(n, 3, 30)
    cats = [f"Cat {i+1}" for i in range(n)]
    colors = cm.rainbow(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    wedges, texts, autotexts = ax.pie(
        data, labels=cats, autopct='%1.1f%%', colors=colors,
        startangle=90, textprops={'fontsize': 7}
    )
    for at in autotexts:
        at.set_fontsize(5)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    return fig, ["too_many_categories", "inappropriate_chart_type", "rainbow_palette", "low_readability"]

def bad_spaghetti_line(title):
    """Line chart with too many overlapping lines (spaghetti chart)."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    n_lines = random.randint(8, 15)
    x = np.arange(12)
    colors = cm.rainbow(np.linspace(0, 1, n_lines))
    for i in range(n_lines):
        y = np.cumsum(np.random.randn(12) * 5 + 2) + random.randint(20, 80)
        ax.plot(x, y, color=colors[i], linewidth=1.5, label=f"Series {i+1}")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=6, ncol=3, loc='upper left')
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    fig.tight_layout()
    return fig, ["spaghetti_chart", "too_many_series", "rainbow_palette", "low_readability"]

def bad_dual_axis_abuse(title):
    """Dual y-axis chart where scales are manipulated to suggest false correlation."""
    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor='white')
    x = np.arange(8)
    cats = random_categories(8)
    y1 = np.array(random_data(8, 100, 500))
    y2 = np.random.randn(8) * 2 + 10  # Totally different scale

    ax1.bar(x, y1, color='#4C72B0', alpha=0.7, width=0.4, label='Revenue ($)')
    ax1.set_ylabel("Revenue ($)", fontsize=10, color='#4C72B0')
    ax1.set_ylim(min(y1) * 0.8, max(y1) * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color='#C44E52', linewidth=2.5, marker='s', markersize=6, label='Satisfaction')
    ax2.set_ylabel("Satisfaction Score", fontsize=10, color='#C44E52')
    ax2.set_ylim(min(y2) - 5, max(y2) + 5)  # Manipulated scale

    # when setting tick labels, ensure locator/labels align
    labels = cats[:8]
    # ensure there are labels before setting ticks
    if not labels:
        return fig, []
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig, ["dual_axis_abuse", "misleading_correlation", "scale_manipulation"]

def bad_heavy_gridlines(data, cats, title):
    """Chart with excessively heavy gridlines that overpower the data."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#f0f0f0')
    # Ensure lengths match
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return fig, []
    ax.set_facecolor('#e0e0e0')
    ax.bar(cats, data, color='#aaaaaa', edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(True, linewidth=2.5, color='black', alpha=0.6)
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    ax.set_ylim(0, max(data) * 1.15)
    fig.tight_layout()
    return fig, ["heavy_gridlines", "low_data_ink_ratio", "data_obscured"]

def bad_rainbow_explosion(data, cats, title):
    """Bar chart with unnecessary rainbow coloring and gradient background."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # Ensure lengths match
    data, cats = _ensure_matching_lengths(data, cats)
    if not data:
        return fig, ["rainbow_palette"]
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    colors = cm.rainbow(np.linspace(0, 1, len(data)))
    bars = ax.bar(cats, data, color=colors, edgecolor='white', linewidth=2, width=0.6)
    # Add glow-like effect
    for bar, c in zip(bars, colors):
        bar.set_alpha(0.85)
    ax.set_title(title, fontsize=15, fontweight='bold', color='white', pad=15)
    ax.set_ylabel("Value", fontsize=11, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)
    ax.grid(True, linewidth=0.5, color='white', alpha=0.2)
    ax.set_ylim(0, max(data) * 1.2)
    fig.tight_layout()
    return fig, ["rainbow_palette", "dark_theme_abuse", "low_data_ink_ratio", "unnecessary_color"]


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_sample(idx, output_dir):
    """Generate one good + one bad chart pair with metadata."""
    n_cats = random.choice([4, 5, 6, 7])
    data = random_data(n_cats)
    cats = random_categories(n_cats)
    title = random_title()
    dates, time_vals = random_time_series()

    # --- Generate GOOD chart ---
    good_type = random.choice(["bar", "line", "scatter", "hbar", "pie"])
    if good_type == "bar":
        fig_good = good_bar_chart(data, cats, title)
    elif good_type == "line":
        fig_good = good_line_chart(dates, time_vals, title)
    elif good_type == "scatter":
        fig_good = good_scatter_chart(title)
    elif good_type == "hbar":
        fig_good = good_horizontal_bar(data, cats, title)
    else:
        fig_good = good_pie_chart_simple(title)

    good_path = os.path.join(output_dir, "good", f"good_{idx:04d}.png")
    save_chart(fig_good, good_path)

    good_meta = {
        "id": f"good_{idx:04d}",
        "label": "good",
        "chart_type": good_type,
        "title": title,
        "violations": [],
        "tufte_score": 1.0,
    }

    # --- Generate BAD chart ---
    bad_choice = random.choice([
        "chartjunk", "3d", "truncated", "pie_many",
        "spaghetti", "dual_axis", "heavy_grid", "rainbow"
    ])

    if bad_choice == "chartjunk":
        fig_bad, violations = bad_chartjunk_bar(data, cats, title)
    elif bad_choice == "3d":
        fig_bad, violations = bad_3d_bar(data, cats, title)
    elif bad_choice == "truncated":
        fig_bad, violations = bad_truncated_axis(data, cats, title)
    elif bad_choice == "pie_many":
        fig_bad, violations = bad_pie_many_categories(title)
    elif bad_choice == "spaghetti":
        fig_bad, violations = bad_spaghetti_line(title)
    elif bad_choice == "dual_axis":
        fig_bad, violations = bad_dual_axis_abuse(title)
    elif bad_choice == "heavy_grid":
        fig_bad, violations = bad_heavy_gridlines(data, cats, title)
    else:
        fig_bad, violations = bad_rainbow_explosion(data, cats, title)

    bad_path = os.path.join(output_dir, "bad", f"bad_{idx:04d}.png")
    save_chart(fig_bad, bad_path)

    bad_meta = {
        "id": f"bad_{idx:04d}",
        "label": "bad",
        "chart_type": bad_choice,
        "title": title,
        "violations": violations,
        "tufte_score": 0.0,
    }

    return good_meta, bad_meta


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic visualization dataset")
    parser.add_argument("--output_dir", type=str, default="./vis_dataset",
                        help="Output directory for generated images")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of good/bad pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create directories
    os.makedirs(os.path.join(args.output_dir, "good"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "bad"), exist_ok=True)

    all_meta = []
    print(f"Generating {args.num_samples} good/bad chart pairs...")

    for i in range(args.num_samples):
        good_m, bad_m = generate_sample(i, args.output_dir)
        all_meta.append(good_m)
        all_meta.append(bad_m)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{args.num_samples} pairs ({(i+1)*2} total images)")

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    # Print summary
    violation_counts = {}
    for m in all_meta:
        for v in m["violations"]:
            violation_counts[v] = violation_counts.get(v, 0) + 1

    print(f"\nDone! Generated {len(all_meta)} total images.")
    print(f"  Good: {sum(1 for m in all_meta if m['label'] == 'good')}")
    print(f"  Bad:  {sum(1 for m in all_meta if m['label'] == 'bad')}")
    print(f"\nViolation distribution:")
    for v, c in sorted(violation_counts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {c}")
    print(f"\nMetadata saved to: {meta_path}")
    print(f"Images saved to: {args.output_dir}/good/ and {args.output_dir}/bad/")


if __name__ == "__main__":
    main()