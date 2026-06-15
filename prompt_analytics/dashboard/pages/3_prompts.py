"""Prompts dashboard page: category / complexity / cost distributions.

Gated behind the ``categorization`` feature flag and behind the presence of
actually-categorized rows (the flag may be on while nothing is classified yet).

Migrated to Apache ECharts (``docs/MIGRATION-ECHARTS.md``). The *category*
dimension is a global filter, so every category-bearing chart is a cross-filter
**emitter** (clicking a category narrows the dashboard to it -> ``KEY_CATEGORIES``):
the twin count/cost bars and the per-category cost box plot. Complexity is not a
filter dimension, so the complexity charts stay read-only. Per-bar labels use a
real :func:`echarts.js` formatter (a bare function string in an option renders
literally; only ``echarts.js``/``events`` JS is executed). Zero Plotly remains.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics.dashboard import data, echarts, filters, theme


def _has_category(series: pd.Series) -> pd.Series:
    """Boolean mask of rows with a non-null, non-empty category value."""
    s = series.astype("object")
    return s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.lower() != "nan")


def _category_summary(cat_prompts: pd.DataFrame, cost: str) -> pd.DataFrame:
    """Per-category indicators: prompt count, total cost, median cost, mean cost."""
    grouped = cat_prompts.groupby("category")
    summary = pd.DataFrame(
        {
            "prompts": grouped.size(),
            "total_cost": grouped[cost].sum(),
            "median_cost": grouped[cost].median(),
            "avg_cost": grouped[cost].mean(),
        }
    )
    result: pd.DataFrame = summary.reset_index()
    result["share"] = 100 * result["prompts"] / result["prompts"].sum()
    return result


def _count_option(summary: pd.DataFrame, order: list[str]) -> dict[str, Any]:
    """Horizontal bar of prompt counts (share % in the label); emits KEY_CATEGORIES."""
    s = summary.set_index("category").reindex(order)
    values = [int(v) for v in s["prompts"].tolist()]
    shares = [round(float(v), 1) for v in s["share"].tolist()]
    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": "Prompts per category",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 8, "right": 80, "top": 48, "bottom": 24, "containLabel": True}
    option["tooltip"].update({"trigger": "item"})
    xaxis = echarts.value_axis()
    xaxis["max"] = int(max(values) * 1.25) + 1 if values else 1
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.category_axis(order, inverse=True)
    option["series"] = [
        {
            "type": "bar",
            "data": [
                {"value": v, "itemStyle": {"color": theme.CATEGORY_COLORS.get(cat, "#9CA3AF")}}
                for cat, v in zip(order, values, strict=True)
            ],
            "itemStyle": {"borderRadius": [0, 4, 4, 0]},
            "label": {
                "show": True,
                "position": "right",
                "color": c["text"],
                "formatter": echarts.js(
                    "function(p){var S=" + json.dumps(shares) + ";"
                    "return Number(p.value).toLocaleString()+'  ('+Number(S[p.dataIndex]).toFixed(0)+'%)';}"
                ),
            },
        }
    ]
    return option


def _cost_option(summary: pd.DataFrame, primary: str, order: list[str]) -> dict[str, Any]:
    """Horizontal bar of total cost (median $/prompt in the label); emits KEY_CATEGORIES."""
    s = summary.set_index("category").reindex(order)
    totals = [round(float(v), 2) for v in s["total_cost"].tolist()]
    meds = [round(float(v), 4) for v in s["median_cost"].tolist()]
    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": f"Cost per category ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 8, "right": 140, "top": 48, "bottom": 24, "containLabel": True}
    option["tooltip"].update({"trigger": "item"})
    xaxis = echarts.value_axis(money=True)
    xaxis["max"] = round(max(totals) * 1.8, 2) if totals else 1
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.category_axis(order, inverse=True)
    option["series"] = [
        {
            "type": "bar",
            "data": [
                {"value": v, "itemStyle": {"color": theme.CATEGORY_COLORS.get(cat, "#9CA3AF")}}
                for cat, v in zip(order, totals, strict=True)
            ],
            "itemStyle": {"borderRadius": [0, 4, 4, 0]},
            "label": {
                "show": True,
                "position": "right",
                "color": c["text"],
                "formatter": echarts.js(
                    "function(p){var M=" + json.dumps(meds) + ";"
                    "return '$'+Number(p.value).toFixed(2)+'  (med $'+Number(M[p.dataIndex]).toFixed(3)+'/prompt)';}"
                ),
            },
        }
    ]
    return option


def _complexity_option(comp: pd.DataFrame) -> dict[str, Any] | None:
    """Discrete bar of the complexity distribution (1-5); read-only (not a filter dim)."""
    work = comp.copy()
    work["complexity"] = pd.to_numeric(work["complexity"], errors="coerce")
    work = work.dropna(subset=["complexity"])
    if work.empty:
        return None
    vc = (
        work["complexity"]
        .round()
        .astype(int)
        .clip(1, 5)
        .value_counts()
        .reindex([1, 2, 3, 4, 5], fill_value=0)
    )
    c = echarts.colors()
    option = echarts.base_option(color=[theme.PALETTE[4]])
    option["legend"] = {"show": False}
    # No in-chart title: the section subheader already says "Complexity
    # distribution" (and an in-chart title collided with the y-axis name).
    option["grid"] = {"left": 56, "right": 24, "top": 24, "bottom": 40, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["xAxis"] = echarts.category_axis([str(i) for i in vc.index.tolist()])
    option["xAxis"]["name"] = "Complexity"
    option["yAxis"] = echarts.value_axis(name="Prompts")
    option["yAxis"]["nameLocation"] = "middle"
    option["yAxis"]["nameGap"] = 40
    option["series"] = [
        {
            "type": "bar",
            "data": [int(v) for v in vc.tolist()],
            "itemStyle": {"color": theme.PALETTE[4], "borderRadius": [4, 4, 0, 0]},
            "label": {"show": True, "position": "top", "color": c["text"]},
        }
    ]
    return option


def _cost_by_category_box_option(
    cat_prompts: pd.DataFrame, cost: str, primary: str
) -> tuple[dict[str, Any], str] | None:
    """Box plot of per-prompt cost by category (median = center line); emits KEY_CATEGORIES.

    Whiskers at p5/p95 + a clipped y-axis keep the boxes legible despite the long
    cost tail; the count above the cap rides in the returned caption note.
    """
    rows: list[tuple[str, list[float], float]] = []
    groups: list[Any] = []
    for category, group in cat_prompts.groupby("category"):
        arr = group[cost].astype(float).to_numpy()
        if arr.size == 0:
            continue
        stats = data.box_stats(arr)
        rows.append((str(category), stats, stats[2]))
        groups.append(arr)
    if not rows:
        return None
    rows.sort(key=lambda r: r[2])
    cats = [r[0] for r in rows]
    y_max, n_above = data.box_cap(groups, [r[1] for r in rows])
    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": f"Cost of one prompt by category ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 56, "right": 24, "top": 48, "bottom": 64, "containLabel": True}
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "trigger": "item",
    }
    option["xAxis"] = echarts.category_axis(cats)
    yaxis = echarts.value_axis(money=True)
    if y_max is not None:
        yaxis["max"] = y_max
    option["yAxis"] = yaxis
    option["series"] = [
        {
            "type": "boxplot",
            "data": [
                {
                    "value": stats,
                    "itemStyle": {
                        "color": theme._rgba(theme.CATEGORY_COLORS.get(cat, "#9CA3AF"), 0.4),
                        "borderColor": theme.CATEGORY_COLORS.get(cat, "#9CA3AF"),
                    },
                }
                for cat, stats, _ in rows
            ],
        }
    ]
    note = (
        f" · {n_above} prompt(s) above ${y_max:,.2f} (axis clipped)"
        if y_max is not None and n_above
        else ""
    )
    return option, note


def _cost_by_complexity_box_option(
    cat_prompts: pd.DataFrame, cost: str, primary: str
) -> tuple[dict[str, Any], str] | None:
    """Box plot of per-prompt cost by complexity; read-only (not a filter dim).

    Same robust-whisker + clipped-axis treatment as the category box; the count
    above the cap is returned as a caption note.
    """
    work = cat_prompts.copy()
    work["complexity"] = pd.to_numeric(work["complexity"], errors="coerce")
    work = work.dropna(subset=["complexity"])
    if work.empty:
        return None
    work["complexity"] = work["complexity"].round().astype(int).clip(1, 5)
    rows: list[tuple[str, list[float]]] = []
    groups: list[Any] = []
    for level in [1, 2, 3, 4, 5]:
        arr = work.loc[work["complexity"] == level, cost].astype(float).to_numpy()
        if arr.size == 0:
            continue
        rows.append((str(level), data.box_stats(arr)))
        groups.append(arr)
    if not rows:
        return None
    y_max, n_above = data.box_cap(groups, [stats for _, stats in rows])
    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": f"Cost of one prompt by complexity ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 56, "right": 24, "top": 48, "bottom": 40, "containLabel": True}
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "trigger": "item",
    }
    option["xAxis"] = echarts.category_axis([r[0] for r in rows])
    option["xAxis"]["name"] = "Complexity"
    yaxis = echarts.value_axis(money=True)
    if y_max is not None:
        yaxis["max"] = y_max
    option["yAxis"] = yaxis
    option["series"] = [
        {
            "type": "boxplot",
            "data": [stats for _, stats in rows],
            "itemStyle": {
                "color": theme._rgba(theme.PALETTE[3], 0.4),
                "borderColor": theme.PALETTE[3],
            },
        }
    ]
    note = (
        f" · {n_above} prompt(s) above ${y_max:,.2f} (axis clipped)"
        if y_max is not None and n_above
        else ""
    )
    return option, note


def main() -> None:
    """Render the Prompts page (gated)."""
    from prompt_analytics.dashboard import echarts as ec

    st.title("Prompts")

    cfg = data.load_config()
    if not cfg.get("features", {}).get("categorization"):
        st.info(
            "Categorization not enabled. Run `prompt-analytics categorize` to enable this page."
        )
        st.stop()

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all)
    prompts = frames.get("prompts", pd.DataFrame())
    primary = data.primary_provider()
    cost = data.cost_col(primary)

    if prompts.empty or "category" not in prompts.columns:
        st.info("No data for the current filters.")
        st.stop()

    cat_prompts = prompts[_has_category(prompts["category"])].copy()
    if cat_prompts.empty:
        st.info(
            "No prompts are categorized yet. Run `prompt-analytics categorize` to "
            "populate category / complexity, then revisit this page."
        )
        st.stop()

    if cost not in cat_prompts.columns:
        cat_prompts[cost] = 0.0
    cat_prompts[cost] = cat_prompts[cost].fillna(0.0)

    st.caption("👆 Click a category on a bar or box to filter the whole dashboard to it.")

    # Per-category indicators: counts + total/median cost side by side. A single
    # shared order (largest count at the top, both charts) lets the eye compare.
    summary = _category_summary(cat_prompts, cost)
    order = summary.sort_values("prompts", ascending=False)["category"].tolist()
    n_total = len(cat_prompts)
    left, right = st.columns(2)
    with left:
        clicked = ec.render(
            _count_option(summary, order), key="prompts_count", height="360px", click=True
        )
        ec.apply_click(clicked, filters.KEY_CATEGORIES)
    with right:
        clicked = ec.render(
            _cost_option(summary, primary, order), key="prompts_cost", height="360px", click=True
        )
        ec.apply_click(clicked, filters.KEY_CATEGORIES)
    st.caption(f"n = {n_total:,} categorized prompts")

    # Complexity distribution (discrete bars 1-5); read-only.
    st.subheader("Complexity distribution")
    if "complexity" in cat_prompts.columns and cat_prompts["complexity"].notna().any():
        comp_opt = _complexity_option(cat_prompts)
        if comp_opt is None:
            st.info("No complexity values available.")
        else:
            ec.render(comp_opt, key="prompts_complexity", height="320px")
            n_comp = int(pd.to_numeric(cat_prompts["complexity"], errors="coerce").notna().sum())
            st.caption(f"n = {n_comp:,} prompts with complexity scores")
    else:
        st.info("No complexity values available.")

    # Distribution of what a single prompt costs.
    st.subheader("Cost of a single prompt")
    st.caption(
        "Each box is the spread of what *individual* prompts cost (not the "
        "category total): the line is the median, the box the middle 50%. The "
        "y-axis is clipped just above the tallest box so a few categories with very "
        "expensive prompts don't flatten everything — their whiskers run off the "
        "top and the count beyond is noted below. A tall box means costs vary a lot."
    )
    box_cat = _cost_by_category_box_option(cat_prompts, cost, primary)
    if box_cat is not None:
        option, clip_note = box_cat
        clicked = ec.render(option, key="prompts_cost_box", height="360px", click=True)
        ec.apply_click(clicked, filters.KEY_CATEGORIES)
        st.caption(
            f"n = {n_total:,} prompts across "
            f"{cat_prompts['category'].nunique()} categories{clip_note}"
        )

    if "complexity" in cat_prompts.columns and cat_prompts["complexity"].notna().any():
        box_comp = _cost_by_complexity_box_option(cat_prompts, cost, primary)
        if box_comp is not None:
            option, clip_note = box_comp
            ec.render(option, key="prompts_complexity_box", height="320px")
            n_comp = int(pd.to_numeric(cat_prompts["complexity"], errors="coerce").notna().sum())
            st.caption(f"n = {n_comp:,} prompts with complexity scores{clip_note}")

    st.caption(
        "Cost vs the prompt's position in its session has its own page — see "
        "**Session depth** for the distribution by depth band."
    )


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
