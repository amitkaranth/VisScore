"""Chart type registry and dispatcher."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from matplotlib.figure import Figure

from visscore_synthetic import non_tufte_charts as nt
from visscore_synthetic import tufte_charts as tt

ChartFn = Callable[[np.random.Generator, int, int, float], tuple[Figure, dict[str, Any]]]

TUFE_CHARTS: dict[str, ChartFn] = {
    "line": tt.chart_tufte_line,
    "scatter": tt.chart_tufte_scatter,
    "bar_horizontal": tt.chart_tufte_bar_horizontal,
    "dot_strip": tt.chart_tufte_dot_strip,
    "small_multiples": tt.chart_tufte_small_multiples,
    "box": tt.chart_tufte_box,
    "sparkline": tt.chart_tufte_sparkline,
    "area": tt.chart_tufte_area,
}

NON_TUFE_CHARTS: dict[str, ChartFn] = {
    "bar_rainbow": nt.chart_non_tufte_bar_rainbow,
    "line_clutter": nt.chart_non_tufte_line_clutter,
    "pie_exploded": nt.chart_non_tufte_pie_exploded,
    "dashboard": nt.chart_non_tufte_dashboard,
    "bar3d": nt.chart_non_tufte_bar3d,
    "scatter_annotated": nt.chart_non_tufte_scatter_annotated,
    "histogram_busy": nt.chart_non_tufte_histogram_busy,
}


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
) -> tuple[Figure, dict[str, Any]]:
    reg = TUFE_CHARTS if class_label == "tufte" else NON_TUFE_CHARTS
    names = list(reg.keys())
    if allowed is not None:
        names = [n for n in names if n in allowed]
    if not names:
        raise ValueError(f"No chart types available for class {class_label!r} after filtering.")
    key = str(rng.choice(names))
    fig, meta = reg[key](rng, width_px, height_px, dpi)
    meta["chart_type"] = key
    meta["class_label"] = class_label
    return fig, meta


def all_tufte_keys() -> list[str]:
    return sorted(TUFE_CHARTS.keys())


def all_non_tufte_keys() -> list[str]:
    return sorted(NON_TUFE_CHARTS.keys())
