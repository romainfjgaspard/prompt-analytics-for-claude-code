"""Composition page: where the cost goes, *by content* (the product's spine).

The dashboard now has a backbone -- input (what you asked), output (what Claude
produced), context (what fills the cache it re-reads). This page is the narrated
home of that spine. It ships the **OUTPUT** section first (Axe C): the language
mix, the code-vs-tests split, the estimated cost per language, and the prose-vs-
code split of the generated tokens -- the differentiator cc-lens never filled in
(its ``languages`` field stays empty) and never priced.

It is a *view* over :func:`analytics.output_composition` (the same numbers the
``by-output`` CLI prints), narrowed to the global sidebar / chart-click selection
via :func:`analytics.filter_prompt_ids` so it honours the same filter as every
other tab. Read-only: language is not a global filter dimension, so the charts
emit no cross-filter. Metrics only -- no source code is ever read here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics import analytics
from prompt_analytics.dashboard import data, echarts, filters, theme

# How many languages to show before folding the long tail into an "Other" slice.
_MIX_TOP = 12
_COST_TOP = 10


def _fold_other(
    pairs: list[tuple[str, float]], top: int, other_label: str = "Other languages"
) -> list[tuple[str, float]]:
    """Keep the ``top`` largest pairs (by value), summing the rest into one bucket."""
    ordered = sorted(pairs, key=lambda kv: -kv[1])
    head = ordered[:top]
    tail_sum = sum(v for _, v in ordered[top:])
    if tail_sum > 0:
        head.append((other_label, tail_sum))
    return head


def _language_mix_option(comp: analytics.OutputComposition) -> dict[str, Any] | None:
    """Horizontal stacked bar of lines produced per language, split Code vs Tests.

    Answers "language mix" and "code vs tests" in one chart: bar length is the
    lines added in that language, the stack shows how much of it is tests.
    """
    langs = [lng for lng in comp.languages if lng.lines_added > 0]
    if not langs:
        return None
    langs = sorted(langs, key=lambda x: -x.lines_added)[:_MIX_TOP]
    # Largest at the top (inverse category axis): keep the order code-first.
    names = [lng.language for lng in langs]
    code_lines = [int(lng.lines_added - lng.test_added) for lng in langs]
    test_lines = [int(lng.test_added) for lng in langs]

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Language mix — lines written",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "roundRect"}
    option["grid"] = {"left": 8, "right": 56, "top": 48, "bottom": 40, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["xAxis"] = echarts.value_axis(name="Lines added")
    option["yAxis"] = echarts.category_axis(names, inverse=True)
    option["series"] = [
        {
            "name": "Code",
            "type": "bar",
            "stack": "lines",
            "data": code_lines,
            "itemStyle": {"color": theme.PALETTE[1]},
        },
        {
            "name": "Tests",
            "type": "bar",
            "stack": "lines",
            "data": test_lines,
            "itemStyle": {"color": theme.CATEGORY_COLORS["test"], "borderRadius": [0, 4, 4, 0]},
        },
    ]
    return option


def _donut(title: str, points: list[dict[str, Any]], money: bool = True) -> dict[str, Any]:
    """A themed donut with percentage labels (shared by the two cost charts)."""
    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": title,
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    fmt = "{b}: ${c} ({d}%)" if money else "{b}: {c} ({d}%)"
    option["tooltip"] = {
        "trigger": "item",
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "formatter": fmt,
    }
    option["legend"] = {
        "bottom": 0,
        "textStyle": {"color": c["text"]},
        "icon": "roundRect",
        "type": "scroll",
    }
    option["series"] = [
        {
            "type": "pie",
            "radius": ["46%", "72%"],
            "center": ["50%", "48%"],
            "avoidLabelOverlap": True,
            "label": {"show": True, "formatter": "{d}%", "color": c["text"], "fontSize": 11},
            "labelLine": {"length": 6, "length2": 6},
            "data": points,
        }
    ]
    return option


def _cost_by_language_option(comp: analytics.OutputComposition) -> dict[str, Any] | None:
    """Donut of the estimated output spend, by language (code side), + tooling."""
    pairs: list[tuple[str, float]] = [
        (lng.language, round(lng.code_cost, 4)) for lng in comp.languages if lng.code_cost > 0
    ]
    if comp.tooling_cost > 0:
        pairs.append(("(other tooling)", round(comp.tooling_cost, 4)))
    if not pairs:
        return None
    folded = _fold_other(pairs, _COST_TOP)
    color_map = theme.language_color_map([name for name, _ in folded])
    points = [
        {
            "name": name,
            "value": round(value, 2),
            "itemStyle": {"color": color_map.get(name, "#9CA3AF")},
        }
        for name, value in folded
    ]
    return _donut(f"Cost by language ({comp.provider})", points, money=True)


def _prose_vs_code_option(comp: analytics.OutputComposition) -> dict[str, Any] | None:
    """Donut of the generated output spend: prose (explanation) vs code/tooling."""
    if comp.prose_cost <= 0 and comp.code_cost <= 0:
        return None
    points = [
        {
            "name": "Prose / explanation",
            "value": round(comp.prose_cost, 2),
            "itemStyle": {"color": theme.TOKEN_TYPE_COLORS["output"]},
        },
        {
            "name": "Code / tooling",
            "value": round(comp.code_cost, 2),
            "itemStyle": {"color": theme.PALETTE[0]},
        },
    ]
    return _donut(f"Prose vs code ({comp.provider})", points, money=True)


def _render_headline(comp: analytics.OutputComposition) -> None:
    """Four KPI cards summarizing what the assistant produced."""
    gen_cost = comp.prose_cost + comp.code_cost
    code_share = round(100 * comp.code_cost / gen_cost, 1) if gen_cost else 0.0
    test_pct = round(100 * comp.total_test / comp.total_added, 1) if comp.total_added else 0.0

    cols = st.columns(4)
    cols[0].metric(
        "Lines written (+)",
        f"{comp.total_added:,}",
        delta=f"−{comp.total_deleted:,} removed" if comp.total_deleted else None,
        delta_color="off",
    )
    cols[1].metric("Files touched", f"{comp.total_files:,}")
    cols[2].metric("Output that is tests", f"{test_pct:.0f}%")
    cols[3].metric(
        f"Generation cost ({comp.provider})",
        f"${gen_cost:,.2f}",
        delta=f"{code_share:.0f}% code",
        delta_color="off",
    )


def main() -> None:
    """Render the Composition page (output section, Axe C)."""
    st.title("Composition")
    st.caption(
        "Where your cost goes, **by content**: input (what you ask) → output (what "
        "Claude produces) → context (what fills the cache it re-reads). This page is "
        "the **output** view — the input breakdown lives on **Prompts** (categories), "
        "the context breakdown on **Optimize**."
    )

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all)

    prompts = frames.get("prompts", pd.DataFrame())
    primary = data.primary_provider()
    if prompts.empty or "prompt_id" not in prompts.columns:
        st.info("No data for the current filters.")
        st.stop()

    # The output-composition CSVs are not part of the dashboard frames; load the
    # raw dataset and narrow it to the same prompts the global filters kept.
    kept_ids = set(prompts["prompt_id"])
    ds = analytics.filter_prompt_ids(data.load_dataset(), kept_ids)
    comp = analytics.output_composition(ds, primary)

    if not comp.has_data:
        st.info(
            "No output-composition data yet. These metrics (language mix, code vs "
            "tests, cost per language, prose vs code) ship with the latest extractor — "
            "re-run `prompt-analytics extract` to populate them, then revisit this page."
        )
        st.stop()

    theme.section(
        "Output — what Claude produced",
        "The language mix and code/tests split come straight from the file edits "
        "(exact line diffs). The cost split is an estimate: each prompt's real output "
        "cost is prorated by a local tokenizer's prose/code weight, and the code half "
        "is attributed across the languages it edited by line churn.",
    )

    _render_headline(comp)

    mix = _language_mix_option(comp)
    if mix is not None:
        echarts.render(mix, key="comp_language_mix", height="420px")
        st.caption(
            f"{comp.total_added:,} lines added across {comp.total_files:,} files · "
            f"{comp.total_test:,} of them in tests"
            + ("" if len(comp.languages) <= _MIX_TOP else f" · top {_MIX_TOP} languages shown")
        )
    else:
        st.info("No file-edit metrics in range (Claude produced prose only, or read-only tools).")

    left, right = st.columns(2)
    with left:
        cost_lang = _cost_by_language_option(comp)
        if cost_lang is not None:
            echarts.render(cost_lang, key="comp_cost_by_language", height="380px")
            st.caption("Estimated output spend attributed to each language (code side).")
        else:
            st.info("No code spend to attribute to a language in range.")
    with right:
        prose_code = _prose_vs_code_option(comp)
        if prose_code is not None:
            echarts.render(prose_code, key="comp_prose_vs_code", height="380px")
            st.caption(
                f"{comp.prose_tokens:,} prose vs {comp.code_tokens:,} code/tool output tokens."
            )
        else:
            st.info("No generated-output tokens in range.")

    st.caption("👉 The same breakdown on the command line: `prompt-analytics by-output`.")


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
