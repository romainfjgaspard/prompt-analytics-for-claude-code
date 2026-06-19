"""Session depth: what a prompt costs — and what it's made of — as a session deepens.

The showcase of prompt-level positioning (8.4). Two linked stories, read top to
bottom:

1. **Cost** — a box plot of what *one* prompt costs at each depth band (median
   line, box = middle 50%, whiskers = p5–p95; the y-axis is clipped just above
   the tallest box so the boxes stay legible). Averages hide the variability that
   makes this interesting; a headline states how a deep prompt compares to an
   opener.
2. **Why** — the median input-side token mix per prompt by depth, one line per
   component: cache *reads* climb (later prompts ride the context) while fresh
   input and cache *writes* shrink. That's the mechanism behind the cost curve.

Migrated to Apache ECharts (``docs/MIGRATION-ECHARTS.md``). Neither chart emits a
cross-filter (depth is not a global filter dimension). Zero Plotly remains.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics.dashboard import data, echarts, filters, theme

# Finer than the CLI table: most prompts live in the deep tail, so the early
# depths get their own band and the tail is split rather than lumped into "11+".
_BANDS: tuple[tuple[int, int | None, str], ...] = (
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, 5, "5"),
    (6, 6, "6"),
    (7, 7, "7"),
    (8, 8, "8"),
    (9, 9, "9"),
    (10, 10, "10"),
    (11, 15, "11-15"),
    (16, 20, "16-20"),
    (21, None, "21+"),
)
_BAND_ORDER = [label for _, _, label in _BANDS]

# Input-side token components, in stack/legend order (colors from theme.MIX_COLORS).
_COMPONENTS = {
    "input": "Fresh input",
    "cache_read": "Cache read",
    "cache_write_5m": "Cache write (5m)",
    "cache_write_1h": "Cache write (1h)",
}


def _band_label(index: float) -> str:
    for lo, hi, label in _BANDS:
        if index >= lo and (hi is None or index <= hi):
            return label
    return _BANDS[-1][2]


def _banded_prompts(prompts: pd.DataFrame, cost_col: str) -> pd.DataFrame:
    """Per-prompt rows with a depth-band column (real prompts only)."""
    work: pd.DataFrame = prompts.dropna(subset=["prompt_index"]).copy()
    work["prompt_index"] = pd.to_numeric(work["prompt_index"], errors="coerce")
    work = work[work["prompt_index"] >= 1]
    work[cost_col] = work[cost_col].fillna(0.0)
    work["band"] = work["prompt_index"].map(_band_label)
    return work


def _cost_box_option(
    banded: pd.DataFrame, cost_col: str, primary: str
) -> tuple[dict[str, Any], str] | None:
    """Box plot of per-prompt cost per depth band (median = center line).

    ``n`` per band rides in the x-axis tick labels so the reader can judge how
    many observations underlie each distribution. Whiskers at p5/p95 + a clipped
    y-axis keep the boxes legible despite the long cost tail; the count above the
    cap is returned as a caption note. Read-only (depth is not a global filter
    dimension).
    """
    rows: list[tuple[str, list[float]]] = []
    groups: list[Any] = []
    n_by_band: dict[str, int] = {}
    for label in _BAND_ORDER:
        arr = banded.loc[banded["band"] == label, cost_col].astype(float).to_numpy()
        n_by_band[label] = int(arr.size)
        if arr.size == 0:
            continue
        rows.append((label, data.box_stats(arr)))
        groups.append(arr)
    if not rows:
        return None
    cats = [r[0] for r in rows]
    y_max, n_above = data.box_cap(groups, [stats for _, stats in rows])
    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": f"Cost of one prompt by session depth ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 72, "right": 24, "top": 56, "bottom": 64, "containLabel": True}
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "trigger": "item",
    }
    xaxis = echarts.category_axis(cats)
    xaxis["name"] = "Depth (prompt index in session)"
    xaxis["nameLocation"] = "middle"
    xaxis["nameGap"] = 40
    xaxis["nameTextStyle"] = {"color": c["muted"]}
    xaxis["axisLabel"]["formatter"] = echarts.js(
        "function(v){var N=" + json.dumps(n_by_band) + ";return v+'\\nn='+(N[v]||0);}"
    )
    option["xAxis"] = xaxis
    yaxis = echarts.value_axis(money=True, name="Cost / prompt (USD)")
    yaxis["nameLocation"] = "middle"
    yaxis["nameGap"] = 56
    if y_max is not None:
        yaxis["max"] = y_max
    option["yAxis"] = yaxis
    option["series"] = [
        {
            "type": "boxplot",
            "data": [stats for _, stats in rows],
            "itemStyle": {
                "color": theme._rgba(theme.PALETTE[0], 0.4),
                "borderColor": theme.PALETTE[0],
            },
        }
    ]
    note = (
        f" · {n_above} prompt(s) above ${y_max:,.2f} (axis clipped)"
        if y_max is not None and n_above
        else ""
    )
    return option, note


def _token_evolution(tokens: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-prompt input-side token quantiles by depth band, per component.

    Sums each prompt's tokens per component (zero-filled so a prompt missing a
    component still counts as 0), then takes the **median and the p25/p75** across
    prompts in the band -- a single summed line would hide the wide inter-session
    spread. The zero-fill is what makes the *decline* of cache writes visible: a
    deep prompt that no longer pays a cache write contributes a 0, not nothing.
    Returns ``{component_label: band-indexed frame[med, p25, p75, n]}``, ``{}`` if
    unavailable. ``n`` is the prompt count behind each band's median (identical
    across components, since every prompt is zero-filled into each one) so the
    chart can show how many prompts each point summarizes.
    """
    if not {"prompt_id", "prompt_index", "token_type", "token_count"} <= set(tokens.columns):
        return {}
    work = tokens.dropna(subset=["prompt_index"]).copy()
    work["prompt_index"] = pd.to_numeric(work["prompt_index"], errors="coerce")
    work = work[work["prompt_index"] >= 1]
    work = work[work["token_type"].isin(_COMPONENTS)]
    if work.empty:
        return {}
    work["component"] = work["token_type"].map(_COMPONENTS)
    work["band"] = work["prompt_index"].map(_band_label)
    per_prompt = work.pivot_table(
        index=["prompt_id", "band"],
        columns="component",
        values="token_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    bands = [b for b in _BAND_ORDER if b in set(per_prompt["band"])]
    grouped = per_prompt.groupby("band")
    sizes = grouped.size()  # prompts per band, shared by every component
    out: dict[str, pd.DataFrame] = {}
    for comp in _COMPONENTS.values():
        if comp not in per_prompt.columns:
            continue
        df = pd.DataFrame(
            {
                "med": grouped[comp].median(),
                "p25": grouped[comp].quantile(0.25),
                "p75": grouped[comp].quantile(0.75),
                "n": sizes,
            }
        ).reindex(bands)
        out[comp] = df
    return out


def _col(values: Any) -> list[float | None]:
    """A list with NaN -> None (ECharts gaps) and values rounded to whole tokens."""
    return [None if pd.isna(v) else round(float(v), 0) for v in values]


def _evolution_small_option(comp: str, df: pd.DataFrame, color: str) -> dict[str, Any]:
    """One small-multiple: median tokens/prompt by depth + a shaded p25-p75 band.

    The band is drawn with the ECharts stacked-area trick: an invisible baseline
    at p25 plus a (p75-p25) area on top of it, both in their own stack group.

    Like the box plot above, ``n`` (prompts behind each depth's median) rides in
    the x-axis tick labels, so a median over a thin tail band reads as such.
    """
    bands = df.index.tolist()
    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": comp,
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 14, "fontWeight": 600},
    }
    option["grid"] = {"left": 56, "right": 16, "top": 40, "bottom": 32, "containLabel": True}
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "trigger": "axis",
    }
    xaxis = echarts.category_axis([str(b) for b in bands])
    xaxis["boundaryGap"] = False
    n_by_band = {str(b): (0 if pd.isna(n) else int(n)) for b, n in zip(bands, df["n"], strict=True)}
    xaxis["axisLabel"]["formatter"] = echarts.js(
        "function(v){var N=" + json.dumps(n_by_band) + ";return v+'\\nn='+(N[v]||0);}"
    )
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.value_axis()
    delta = [
        None if pd.isna(lo) or pd.isna(hi) else round(float(hi - lo), 0)
        for lo, hi in zip(df["p25"], df["p75"], strict=True)
    ]
    invisible = {"opacity": 0}
    option["series"] = [
        {
            "type": "line",
            "name": "p25",
            "data": _col(df["p25"]),
            "stack": "band",
            "symbol": "none",
            "lineStyle": invisible,
            "silent": True,
        },
        {
            "type": "line",
            "name": "p25–p75",
            "data": delta,
            "stack": "band",
            "symbol": "none",
            "lineStyle": invisible,
            "areaStyle": {"color": theme._rgba(color, 0.16)},
            "silent": True,
        },
        {
            "type": "line",
            "name": "median",
            "data": _col(df["med"]),
            "smooth": True,
            "showSymbol": True,
            "symbolSize": 5,
            "lineStyle": {"color": color, "width": 2},
            "itemStyle": {"color": color},
        },
    ]
    return option


def main() -> None:
    """Render the Session depth page."""
    from prompt_analytics.dashboard import echarts as ec

    st.title("Session depth")

    st.caption(
        "How much does a prompt actually cost once a session gets deep — and why? "
        "Session openers carry the framing context; later prompts ride the cache. "
        "The box plot shows the full cost distribution at each depth; the lines "
        "below show what each prompt is made of as the session grows."
    )

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all)
    prompts = frames.get("prompts", pd.DataFrame())
    tokens = frames.get("tokens", pd.DataFrame())
    primary = data.primary_provider()
    cost_col = data.cost_col(primary)

    if prompts.empty or "prompt_index" not in prompts.columns or cost_col not in prompts.columns:
        st.info("No data — run `prompt-analytics extract`.")
        st.stop()

    banded = _banded_prompts(prompts, cost_col)
    if banded.empty:
        st.info("No depth data available.")
        st.stop()

    # Headline: a deep prompt vs a session opener, on the median.
    med_cost = banded.groupby("band")[cost_col].median()
    present = [b for b in _BAND_ORDER if b in med_cost.index]
    opener = float(med_cost.get("1", np.nan))
    deepest = present[-1]
    deep_med = float(med_cost.get(deepest, np.nan))
    cols = st.columns(3)
    cols[0].metric("Median cost, session opener", f"${opener:,.2f}")
    cols[1].metric(f"Median cost, depth {deepest}", f"${deep_med:,.2f}")
    if opener:
        cols[2].metric("Deep prompt vs opener", f"x{deep_med / opener:,.2f}")

    box_opt = _cost_box_option(banded, cost_col, primary)
    if box_opt is not None:
        option, clip_note = box_opt
        ec.render(option, key="depth_cost_box", height="420px")
        st.caption(
            f"n = {len(banded):,} prompts across {banded['band'].nunique()} depth bands{clip_note}"
        )

    evo = _token_evolution(tokens)
    if evo:
        st.subheader("What one prompt is made of, by depth")
        st.caption(
            "Median input-side tokens per prompt at each depth; the shaded band "
            "is the p25–p75 spread (sessions vary a lot, so a single line would "
            "mislead). Each panel has its own scale; **n** under each depth is "
            "how many prompts that median summarizes."
        )
        comps = [comp for comp in _COMPONENTS.values() if comp in evo]
        grid = [st.columns(2), st.columns(2)]
        for i, comp in enumerate(comps):
            col = grid[i // 2][i % 2]
            with col:
                ec.render(
                    _evolution_small_option(comp, evo[comp], theme.MIX_COLORS[comp]),
                    key=f"depth_evo_{i}",
                    height="240px",
                )
        # Data-driven takeaway: how the make-up shifts opener -> deepest.
        read = "Cache read"
        if read in evo and len(evo[read]) > 1:
            r0 = float(evo[read]["med"].iloc[0])
            r1 = float(evo[read]["med"].iloc[-1])
            if r0 > 0 and r1 > r0:
                st.caption(
                    f"Cache read takes over: the median prompt reads "
                    f"x{r1 / r0:,.1f} more cached context at depth {deepest} than "
                    "at the opener, while fresh input and cache writes shrink — "
                    "that's why a deep prompt costs less. Costs include sub-agent "
                    "usage rolled into the parent prompt."
                )


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
