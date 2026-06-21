"""Shared in-cell colour scaling for the Explorer tables.

The Prompt Explorer and the File Explorer show the same kinds of magnitude --
plain **counts** (edits, lines, reads, prompts, chars) and **costs** (context $,
load, rent, prompt cost). Both shade each cell by its magnitude (a per-column
heat scale), counts in blue and costs in coral, so the two pages read identically
and the eye finds the big numbers without reading every one.

A per-cell ``background-color`` is used (not ``Styler.bar``): ``st.dataframe``
renders a Styler's solid cell colours but *not* its bar gradients, and the colour
scale needs no extra dependency (no matplotlib).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from functools import partial
from typing import Any

import pandas as pd
from pandas.io.formats.style import Styler

from prompt_analytics.dashboard import theme

# Alpha floor/ceiling for the heat shade: even the smallest non-zero value keeps a
# faint tint, the column max reaches the strongest (but never opaque, so the text
# stays readable on the dark grid).
_ALPHA_MIN, _ALPHA_MAX = 0.10, 0.62


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _heat(series: pd.Series, rgb: tuple[int, int, int]) -> list[str]:
    """Per-cell ``background-color`` for one column: alpha scales with the value."""
    values = pd.to_numeric(series, errors="coerce")
    top = float(values.max()) if values.notna().any() else 0.0
    css: list[str] = []
    for value in values:
        if pd.isna(value) or top <= 0 or float(value) <= 0:
            css.append("")
            continue
        alpha = _ALPHA_MIN + (_ALPHA_MAX - _ALPHA_MIN) * (float(value) / top)
        css.append(f"background-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.3f})")
    return css


def bar_table(
    df: pd.DataFrame,
    *,
    count_cols: Sequence[str] = (),
    cost_cols: Sequence[str] = (),
) -> Styler:
    """A Styler for ``df`` with magnitude-shaded cells + number formatting.

    ``count_cols`` get a blue heat shade and a thousands-separated integer format;
    ``cost_cols`` get a coral heat shade and a ``$`` money format. Columns absent
    from ``df`` are skipped, and an empty frame is returned formatted but unshaded.
    """
    counts: list[Hashable] = [c for c in count_cols if c in df.columns]
    costs: list[Hashable] = [c for c in cost_cols if c in df.columns]
    fmt: dict[Any, str | Callable[[object], str] | None] = {}
    for col in counts:
        fmt[col] = "{:,.0f}"
    for col in costs:
        fmt[col] = "${:,.2f}"
    styler = df.style.format(fmt)
    if df.empty:
        return styler
    for col in counts:
        styler = styler.apply(partial(_heat, rgb=_rgb(theme.BAR_COUNT_COLOR)), subset=[col])
    for col in costs:
        styler = styler.apply(partial(_heat, rgb=_rgb(theme.BAR_COST_COLOR)), subset=[col])
    return styler
