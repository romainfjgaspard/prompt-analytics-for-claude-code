"""Compare page: before vs after a switch date, in workload-normalized ratios.

The capstone transverse view (Axe E / DASH2). Pick a *switch date* — when you
installed a tool, scoped a CLAUDE.md, changed a model — and the page splits the
whole extracted history on it and reads it as **before (left) vs after (right)**.

Everything here is an **average or a ratio, never a sum**: the five
workload-normalized ratios of :func:`analytics.impact_report` (so the cards match
the CLI ``impact`` table exactly when the window is the full history), the
**composition shift** (prose/code and context loading/rent shares + before/after
donuts, the most direct read of trimming output or context), then two averaged
charts side by side — per-prompt cost by token type, and the output language mix
(a normalized share; the prompt-category mix was dropped as not robust enough for a
before/after read). A raw before/after would be confounded by how much you worked;
the normalized view isolates the config from the workload, with the workload
confounders kept honest in an expander.

A **comparison window** (1 week / 1 month / full history) restricts each side to N
days immediately around the pivot, so the two windows are equal-length — a
like-for-like read of an optimization installed on the pivot day.

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


def _language_mix_option(
    before: analytics.Dataset, after: analytics.Dataset, provider: str
) -> dict[str, Any] | None:
    """Grouped horizontal bar: language **share** (% of code lines), before vs after.

    An output composition view (replacing the prompt-category mix, whose labels are
    not robust enough for a before/after read): which languages the written code
    shifted toward, as a normalized share so a busier side never dominates.
    """
    b = impact.language_share(before, provider)
    a = impact.language_share(after, provider)
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
        "text": "Language mix (% of code lines)",
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


def _prose_code_donut(comp: analytics.OutputComposition, title: str) -> dict[str, Any] | None:
    """A donut of the prose vs code halves of the output spend (one side of the split).

    The same shape as the Composition page's headline donut, re-titled for the
    before/after pair: prose cyan ("what comes out"), code coral.
    """
    prose, code = round(comp.prose_cost, 2), round(comp.code_cost, 2)
    if prose <= 0 and code <= 0:
        return None

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": title,
        "left": "center",
        "textStyle": {"color": c["text"], "fontSize": 14, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "circle"}
    option["tooltip"] = {**option["tooltip"], "trigger": "item", "formatter": "{b}: ${c} ({d}%)"}
    option["series"] = [
        {
            "type": "pie",
            "radius": ["40%", "66%"],
            "center": ["50%", "52%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderColor": c["grid"], "borderWidth": 2},
            "label": {"color": c["text"], "formatter": "{b}\n{d}%"},
            "data": [
                {
                    "value": prose,
                    "name": "Prose",
                    "itemStyle": {"color": theme.TOKEN_TYPE_COLORS["output"]},
                },
                {"value": code, "name": "Code", "itemStyle": {"color": theme.PALETTE[0]}},
            ],
        }
    ]
    return option


def _load_rent_donut(comp: analytics.ContextCost, title: str) -> dict[str, Any] | None:
    """A donut of one-off loading vs rent of the context, in tokens (one side of the split).

    The same shape as the Composition page's context donut, re-titled: loading is a
    cache *write* (purple), rent a cache *read* (green), so the mechanic is legible.
    """
    load, rent = comp.load_tokens, comp.rent_read_tokens
    if load <= 0 and rent <= 0:
        return None

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": title,
        "left": "center",
        "textStyle": {"color": c["text"], "fontSize": 14, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "circle"}
    option["tooltip"] = {**option["tooltip"], "trigger": "item", "formatter": "{b}: {c} ({d}%)"}
    option["series"] = [
        {
            "type": "pie",
            "radius": ["40%", "66%"],
            "center": ["50%", "52%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderColor": c["grid"], "borderWidth": 2},
            "label": {"color": c["text"], "formatter": "{b}\n{d}%"},
            "data": [
                {
                    "value": load,
                    "name": "Loading (one-off)",
                    "itemStyle": {"color": theme.TOKEN_TYPE_COLORS["cache_write_1h"]},
                },
                {
                    "value": rent,
                    "name": "Rent (every turn)",
                    "itemStyle": {"color": theme.TOKEN_TYPE_COLORS["cache_read"]},
                },
            ],
        }
    ]
    return option


def _render_composition_shift(
    before: analytics.Dataset, after: analytics.Dataset, provider: str
) -> None:
    """Composition shift: prose/code & context shares (cards) + before/after donuts.

    The most direct read of *did the optimization land*: shrink the prose, the code
    or the context and these shares move. Shares (ratios), not sums, so a busier side
    never reads as "more". The donuts make the same shift visual and stay laid out
    like the cards above -- a **Before** band (output + context donuts side by side)
    over an **After** band -- so before is always read top, after bottom.
    """
    comp_b = analytics.output_composition(before, provider)
    comp_a = analytics.output_composition(after, provider)
    ctx_b = analytics.context_cost(before, provider)
    ctx_a = analytics.context_cost(after, provider)

    if not (comp_b.has_data or comp_a.has_data or ctx_b.has_data or ctx_a.has_data):
        st.info("No output/context composition on either side of the pivot.")
        return

    impact.render_share_columns(
        [
            (
                "Code share of output",
                impact.output_code_share(comp_b),
                impact.output_code_share(comp_a),
                "pct",
            ),
            (
                "Tests share of code lines",
                impact.output_test_share(comp_b),
                impact.output_test_share(comp_a),
                "pct",
            ),
            (
                "Context rent share",
                impact.context_rent_share(ctx_b),
                impact.context_rent_share(ctx_a),
                "pct",
            ),
            (
                "Tokens / code line",
                impact.output_tokens_per_line(comp_b),
                impact.output_tokens_per_line(comp_a),
                "ratio",
            ),
            (
                "Cost / code line",
                impact.output_cost_per_line(comp_b),
                impact.output_cost_per_line(comp_a),
                "money",
            ),
        ]
    )

    def _band(
        label: str, comp: analytics.OutputComposition, ctx: analytics.ContextCost, suffix: str
    ) -> None:
        """One side's donut band: output (prose/code) and context (load/rent), side by side."""
        st.markdown(label)
        out_col, ctx_col = st.columns(2)
        with out_col:
            opt = _prose_code_donut(comp, "Output — prose vs code")
            if opt is not None:
                echarts.render(opt, key=f"compare_prose_{suffix}", height="320px")
            else:
                st.info("No output spend on this side.")
        with ctx_col:
            opt = _load_rent_donut(ctx, "Context — loading vs rent (tokens)")
            if opt is not None:
                echarts.render(opt, key=f"compare_ctx_{suffix}", height="320px")
            else:
                st.info("No context on this side.")

    _band("**◀ Before**", comp_b, ctx_b, "before")
    _band("**After ▶**", comp_a, ctx_a, "after")

    st.caption(
        "Shares (ratios), not sums — a tool that trims prose, code or context moves these. "
        "**Output** is the prose/code split of generation spend; **context** the one-off "
        "loading vs the rent re-read every turn (the lingering-context cost you cut by "
        "compacting and keeping CLAUDE.md short)."
    )


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

    pivot, window_days = impact.render_pivot_picker(frames)
    if pivot is None:
        st.info("No dates available to split on.")
        st.stop()
    scoped = impact.window_dataset(ds, pivot, window_days)

    report = analytics.impact_report(scoped, provider=primary, pivot=pivot)
    impact.render_summary_caption(report, pivot)
    if window_days is not None:
        st.caption(
            f"⏳ **{window_days}-day window per side**: each side is the {window_days} days "
            "immediately around the pivot, so the two windows are equal-length. (The CLI "
            "`impact` table always uses the full history.)"
        )

    ratio_note = (
        "The five workload-normalized ratios (the same numbers the CLI `impact` prints)."
        if window_days is None
        else "The five workload-normalized ratios, on the selected window around the pivot."
    )
    theme.section("Normalized ratios — before vs after", ratio_note)
    impact.render_ratio_columns(report)

    before, after = analytics.split_on_pivot(scoped, pivot)
    theme.section(
        "Composition shift — before vs after",
        "Did the optimization land? The output/context shares, the per-line efficiency (tokens "
        "and cost to write a line of code), and the donuts before vs after — the most direct "
        "read of trimming output or context.",
    )
    _render_composition_shift(before, after, primary)

    theme.section(
        "Averages — before vs after",
        "Per-prompt cost by token type and the output language mix, so a busier side never reads "
        "as more expensive just for running more.",
    )
    left, right = st.columns(2)
    with left:
        opt = _cost_per_prompt_option(before, after, primary)
        if opt is not None:
            echarts.render(opt, key="compare_cost_per_prompt", height="440px")
        else:
            st.info("No priced tokens on either side of the pivot.")
    with right:
        opt = _language_mix_option(before, after, primary)
        if opt is not None:
            echarts.render(opt, key="compare_language_mix", height="440px")
        else:
            st.info("No code-line output on either side of the pivot.")

    impact.render_confounders(report)
    impact.render_honesty_note()


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
