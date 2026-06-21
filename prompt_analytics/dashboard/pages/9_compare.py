"""Compare page: before vs after a switch date, in workload-normalized ratios.

The capstone transverse view (Axe E / DASH2). Pick a *switch date* — when you
installed a tool, scoped a CLAUDE.md, changed a model — and the page splits the
whole extracted history on it and reads it as **before (left) vs after (right)**.

Everything here is an **average or a ratio, never a sum**: the five
workload-normalized ratios of :func:`analytics.impact_report` (so the cards match
the CLI ``impact`` table exactly), then two averaged charts side by side —
per-prompt cost by token type, and the category mix. A raw before/after would be
confounded by how much you worked; the normalized view isolates the config from
the workload, with the workload confounders kept honest in an expander.

Reads the raw :class:`Dataset` (the full history), so the global sidebar filters
do not apply — stated on the page, the same convention as Optimize. Migrated to
Apache ECharts; the charts emit no cross-filter.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit import runtime

from prompt_analytics import analytics
from prompt_analytics.dashboard import data, echarts, filters, impact, theme
from prompt_analytics.schema import TOKEN_TYPE_LABELS, TOKEN_TYPES

# Top categories to chart in the mix (the rest would be noise on a grouped bar).
_MIX_TOP = 12

# Before / after series colours: a neutral "before" and the brand coral for
# "after", reused across both charts so the two sides read identically.
_BEFORE_COLOR = "#6B7280"  # slate grey
_AFTER_COLOR = theme.PALETTE[0]  # coral


def _cost_per_prompt_option(
    before: analytics.Dataset, after: analytics.Dataset, provider: str
) -> dict[str, Any] | None:
    """Grouped vertical bar: average **cost per prompt** by token type, before vs after."""
    b = impact.token_cost_per_prompt(before, provider)
    a = impact.token_cost_per_prompt(after, provider)
    types = [tt for tt in TOKEN_TYPES if b.get(tt) or a.get(tt)]
    if not types:
        return None
    labels = [TOKEN_TYPE_LABELS[tt] for tt in types]
    before_vals = [round(b.get(tt, 0.0), 4) for tt in types]
    after_vals = [round(a.get(tt, 0.0), 4) for tt in types]

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Cost per prompt, by token type",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "roundRect"}
    option["grid"] = {"left": 56, "right": 24, "top": 56, "bottom": 64, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["tooltip"]["valueFormatter"] = echarts.js(
        "function(v){return '$'+Number(v).toFixed(4);}"
    )
    option["xAxis"] = echarts.category_axis(labels)
    option["xAxis"]["axisLabel"] = {"color": c["text"], "rotate": 30, "fontSize": 10}
    option["yAxis"] = echarts.value_axis(money=True)
    option["series"] = [
        {
            "name": "Before",
            "type": "bar",
            "data": before_vals,
            "itemStyle": {"color": _BEFORE_COLOR},
        },
        {"name": "After", "type": "bar", "data": after_vals, "itemStyle": {"color": _AFTER_COLOR}},
    ]
    return option


def _category_mix_option(
    before: analytics.Dataset, after: analytics.Dataset
) -> dict[str, Any] | None:
    """Grouped horizontal bar: category **share** (% of prompts), before vs after."""
    b = impact.category_share(before)
    a = impact.category_share(after)
    cats = sorted(set(b) | set(a), key=lambda cat: -(a.get(cat, 0.0) + b.get(cat, 0.0)))
    cats = cats[:_MIX_TOP]
    if not cats:
        return None
    # Largest at the top (inverse axis), so reverse for the inverted category axis.
    cats = cats[::-1]
    before_vals = [round(b.get(cat, 0.0), 1) for cat in cats]
    after_vals = [round(a.get(cat, 0.0), 1) for cat in cats]

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Category mix (% of prompts)",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "roundRect"}
    option["grid"] = {"left": 8, "right": 48, "top": 56, "bottom": 48, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["tooltip"]["valueFormatter"] = echarts.js(
        "function(v){return Number(v).toFixed(1)+'%';}"
    )
    xaxis = echarts.value_axis()
    xaxis["axisLabel"] = {"color": c["text"], "formatter": "{value}%"}
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.category_axis(cats)
    option["series"] = [
        {
            "name": "Before",
            "type": "bar",
            "data": before_vals,
            "itemStyle": {"color": _BEFORE_COLOR},
        },
        {"name": "After", "type": "bar", "data": after_vals, "itemStyle": {"color": _AFTER_COLOR}},
    ]
    return option


def main() -> None:
    """Render the Compare page."""
    st.title("Compare")
    # No sidebar filters on this page; keep any selection from other pages alive.
    filters.persist_filters()
    st.caption(
        "Split the **full extracted history** on a switch date and read it as **before vs "
        "after**, in workload-normalized ratios and averages — never raw totals. The global "
        "sidebar filters do not apply on this page."
    )

    frames = data.load_all()
    ds = data.load_dataset()
    if not ds.prompts:
        st.info(
            "No data to compare: re-run `prompt-analytics extract` to produce the history first."
        )
        st.stop()
    primary = data.primary_provider()

    pivot = impact.render_pivot_picker(frames)
    if pivot is None:
        st.info("No dates available to split on.")
        st.stop()

    report = analytics.impact_report(ds, provider=primary, pivot=pivot)
    impact.render_summary_caption(report, pivot)

    theme.section(
        "Normalized ratios — before vs after",
        "The five workload-normalized ratios (the same numbers the CLI `impact` prints).",
    )
    impact.render_ratio_columns(report)

    before, after = analytics.split_on_pivot(ds, pivot)
    theme.section(
        "Averages — before vs after",
        "Per-prompt cost by token type and the category mix, so a busier side never reads as "
        "more expensive just for running more.",
    )
    left, right = st.columns(2)
    with left:
        opt = _cost_per_prompt_option(before, after, primary)
        if opt is not None:
            echarts.render(opt, key="compare_cost_per_prompt", height="440px")
        else:
            st.info("No priced tokens on either side of the pivot.")
    with right:
        opt = _category_mix_option(before, after)
        if opt is not None:
            echarts.render(opt, key="compare_category_mix", height="440px")
        else:
            st.info("No categorized prompts on either side of the pivot.")

    impact.render_confounders(report)
    impact.render_honesty_note()


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
