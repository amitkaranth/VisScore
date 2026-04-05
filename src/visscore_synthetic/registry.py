"""Chart type registry, multi-library merge, and dispatcher."""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from visscore_synthetic import non_tufte_charts as nt
from visscore_synthetic import sns_non_tufte_charts as snt
from visscore_synthetic import sns_tufte_charts as st
from visscore_synthetic import tufte_charts as tt
from visscore_synthetic.mpl_styles import is_seaborn_chart_key, resolve_mpl_style_pool

ChartFn = Callable[[np.random.Generator, int, int, float], tuple[Figure, dict[str, Any]]]

LIB_MATPLOTLIB = "matplotlib"
LIB_SEABORN = "seaborn"

TUFE_MATPLOTLIB: dict[str, ChartFn] = {
    "line": tt.chart_tufte_line,
    "scatter": tt.chart_tufte_scatter,
    "bar_horizontal": tt.chart_tufte_bar_horizontal,
    "dot_strip": tt.chart_tufte_dot_strip,
    "small_multiples": tt.chart_tufte_small_multiples,
    "box": tt.chart_tufte_box,
    "sparkline": tt.chart_tufte_sparkline,
    "area": tt.chart_tufte_area,
}

TUFE_SEABORN: dict[str, ChartFn] = {
    "sns_reg_minimal": st.chart_sns_reg_minimal,
    "sns_kde_1d": st.chart_sns_kde_1d,
    "sns_violin_light": st.chart_sns_violin_light,
    "sns_heatmap_muted": st.chart_sns_heatmap_muted,
    "sns_line_facet_subtle": st.chart_sns_line_facet_subtle,
}

NON_TUFE_MATPLOTLIB: dict[str, ChartFn] = {
    "bar_rainbow": nt.chart_non_tufte_bar_rainbow,
    "line_clutter": nt.chart_non_tufte_line_clutter,
    "pie_exploded": nt.chart_non_tufte_pie_exploded,
    "dashboard": nt.chart_non_tufte_dashboard,
    "bar3d": nt.chart_non_tufte_bar3d,
    "scatter_annotated": nt.chart_non_tufte_scatter_annotated,
    "histogram_busy": nt.chart_non_tufte_histogram_busy,
}

NON_TUFE_SEABORN: dict[str, ChartFn] = {
    "sns_heatmap_loud": snt.chart_sns_heatmap_loud,
    "sns_box_swarm_busy": snt.chart_sns_box_swarm_busy,
    "sns_kde_overlay_loud": snt.chart_sns_kde_overlay_loud,
    "sns_bar_estimator_show": snt.chart_sns_bar_estimator_show,
    "sns_scatter_hue_size": snt.chart_sns_scatter_hue_size,
    "sns_clustermap_busy": snt.chart_sns_clustermap_busy,
}


def parse_libraries(s: str | None) -> frozenset[str]:
    if not s or not s.strip():
        return frozenset({LIB_MATPLOTLIB, LIB_SEABORN})
    parts = {p.strip().lower() for p in s.split(",") if p.strip()}
    allowed = {LIB_MATPLOTLIB, LIB_SEABORN}
    bad = parts - allowed
    if bad:
        raise ValueError(f"Unknown --libraries entries: {sorted(bad)}. Use: matplotlib, seaborn")
    return frozenset(parts)


def build_tufte_registry(libraries: frozenset[str]) -> dict[str, ChartFn]:
    d: dict[str, ChartFn] = {}
    if LIB_MATPLOTLIB in libraries:
        d.update(TUFE_MATPLOTLIB)
    if LIB_SEABORN in libraries:
        d.update(TUFE_SEABORN)
    if not d:
        d.update(TUFE_MATPLOTLIB)
    return d


def build_non_tufte_registry(libraries: frozenset[str]) -> dict[str, ChartFn]:
    d: dict[str, ChartFn] = {}
    if LIB_MATPLOTLIB in libraries:
        d.update(NON_TUFE_MATPLOTLIB)
    if LIB_SEABORN in libraries:
        d.update(NON_TUFE_SEABORN)
    if not d:
        d.update(NON_TUFE_MATPLOTLIB)
    return d


def parse_chart_filter(s: str | None, registry: dict[str, ChartFn]) -> frozenset[str] | None:
    if not s or not s.strip():
        return None
    names = {x.strip() for x in s.split(",") if x.strip()}
    unknown = names - set(registry.keys())
    if unknown:
        raise ValueError(f"Unknown chart keys: {sorted(unknown)}. Valid: {sorted(registry)}")
    return frozenset(names)


def draw_random_chart(
    rng: np.random.Generator,
    class_label: str,
    width_px: int,
    height_px: int,
    dpi: float,
    allowed: frozenset[str] | None,
    registry: dict[str, ChartFn],
    mpl_style_mode: str,
) -> tuple[Figure, dict[str, Any]]:
    names = list(registry.keys())
    if allowed is not None:
        names = [n for n in names if n in allowed]
    if not names:
        raise ValueError(f"No chart types available for class {class_label!r} after filtering.")
    key = str(rng.choice(names))
    fn = registry[key]
    style_pool = resolve_mpl_style_pool(class_label, mpl_style_mode)

    if is_seaborn_chart_key(key) or not style_pool:
        fig, meta = fn(rng, width_px, height_px, dpi)
    else:
        style = str(rng.choice(style_pool))
        with plt.style.context(style):
            fig, meta = fn(rng, width_px, height_px, dpi)
        meta["matplotlib_style"] = style

    meta.setdefault("renderer", "seaborn" if is_seaborn_chart_key(key) else "matplotlib")
    meta["chart_type"] = key
    meta["class_label"] = class_label
    return fig, meta


def all_tufte_keys(libraries: frozenset[str]) -> list[str]:
    return sorted(build_tufte_registry(libraries).keys())


def all_non_tufte_keys(libraries: frozenset[str]) -> list[str]:
    return sorted(build_non_tufte_registry(libraries).keys())
