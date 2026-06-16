"""Models page: how usage, cost and per-session spend split across models.

The subscription-vs-usage cost comparison (Claude plans, GitHub Copilot) lives
on the Quotas page.

Migrated to Apache ECharts (``docs/MIGRATION-ECHARTS.md``). Every view is on the
*model* dimension, so each one is a cross-filter **emitter** — clicking a model
narrows the whole dashboard to it (``filters.KEY_MODELS``). The daily stacked bar
returns ``seriesName`` (the model is the series); the pies and the box plot have
the model on the category/slice and return ``name``. Zero Plotly remains here.

Formatters are ECharts *template* strings (``{b}``/``{c}``/``{d}``/``{value}``),
never JS function strings: ``streamlit-echarts`` only evaluates JS passed through
the ``events`` dict, so a function string left in the option renders literally.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics.dashboard import data, echarts, filters, theme


def _pie_option(
    names: list[str], values: Sequence[float], title: str, *, money: bool
) -> dict[str, Any]:
    """A donut pie over the *model* dimension (a click emits ``KEY_MODELS``).

    ``money`` formats the tooltip value as USD. Slice colors come from the shared
    ``model_color_map`` so a model keeps the same hue across every chart on the
    page; slice ``name`` stays the raw id (color + click key) while labels and
    tooltips show ``theme.model_label`` via formatters.
    """
    c = echarts.colors()
    color_map = theme.model_color_map(names)
    labels_map = {n: theme.model_label(n) for n in names}
    map_js = json.dumps(labels_map)
    val_js = "'$'+Number(p.value).toLocaleString()" if money else "Number(p.value).toLocaleString()"
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option.pop("grid", None)
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "trigger": "item",
        "formatter": echarts.js(
            "function(p){var M=" + map_js + ";var n=M[p.name]!==undefined?M[p.name]:p.name;"
            "return n+': '+(" + val_js + ")+' ('+p.percent.toFixed(0)+'%)';}"
        ),
    }
    option["title"] = {
        "text": title,
        "left": "center",
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["series"] = [
        {
            "type": "pie",
            "radius": ["38%", "70%"],
            "center": ["50%", "55%"],
            "avoidLabelOverlap": True,
            "data": [
                {"value": v, "name": n, "itemStyle": {"color": color_map.get(n, "#9CA3AF")}}
                for n, v in zip(names, values, strict=True)
            ],
            "label": {
                "color": c["text"],
                "formatter": echarts.js(
                    "function(p){var M=" + map_js + ";var n=M[p.name]!==undefined?M[p.name]:p.name;"
                    "return n+'\\n'+p.percent.toFixed(0)+'%';}"
                ),
            },
            "labelLine": {"lineStyle": {"color": c["muted"]}},
            "emphasis": {"itemStyle": {"shadowBlur": 8, "shadowColor": "rgba(0,0,0,0.4)"}},
        }
    ]
    return option


def _tokens_by_model_option(tokens: pd.DataFrame) -> dict[str, Any] | None:
    """Donut of token *volume* by model (server-tool rows excluded, as elsewhere)."""
    work = tokens[tokens["token_type"] != "server_tool_use"].dropna(subset=["model"])
    by_model = work.groupby("model", as_index=False)["token_count"].sum()
    by_model = by_model[by_model["token_count"] > 0].sort_values("token_count", ascending=False)
    if by_model.empty:
        return None
    names = by_model["model"].tolist()
    values = [int(v) for v in by_model["token_count"].tolist()]
    return _pie_option(names, values, "Token volume by model", money=False)


def _cost_by_model_pie_option(tokens: pd.DataFrame, primary: str) -> dict[str, Any] | None:
    """Donut of cost by model (primary provider's grid)."""
    col = data.cost_col(primary)
    by_model = tokens.dropna(subset=["model"]).groupby("model", as_index=False)[col].sum()
    by_model = by_model[by_model[col] > 0].sort_values(col, ascending=False)
    if by_model.empty:
        return None
    names = by_model["model"].tolist()
    values = [round(float(v), 2) for v in by_model[col].tolist()]
    return _pie_option(names, values, f"Cost by model ({primary})", money=True)


def _prompt_cost_distribution_option(
    prompts: pd.DataFrame, primary: str
) -> tuple[dict[str, Any], str] | None:
    """Box-and-whisker of per-**prompt** cost by model (median = middle line).

    Per-*session* cost lives on the Sessions page; here the unit is a single
    prompt, which is the natural grain for comparing models. A click on a box
    emits that model's ``name`` -> ``KEY_MODELS``. Boxes are ordered by median;
    whiskers at p5/p95 + a clipped y-axis keep them legible despite the long
    cost tail; the count above the cap is returned as a caption note.
    """
    col = data.cost_col(primary)
    if prompts.empty or "model" not in prompts.columns or col not in prompts.columns:
        return None
    work = prompts.dropna(subset=["model"]).copy()
    work[col] = work[col].fillna(0.0)
    if work.empty:
        return None

    rows: list[tuple[str, list[float], float]] = []
    groups: list[Any] = []
    for model, group in work.groupby("model"):
        costs = group[col].astype(float).to_numpy()
        if costs.size == 0:
            continue
        stats = data.box_stats(costs)
        rows.append((str(model), stats, stats[2]))
        groups.append(costs)
    if not rows:
        return None
    rows.sort(key=lambda r: r[2])  # by median
    models = [r[0] for r in rows]
    color_map = theme.model_color_map(models)
    y_max, n_above = data.box_cap(groups, [r[1] for r in rows])

    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": f"Per-prompt cost distribution by model ({primary})",
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
    xaxis = echarts.category_axis(models)
    xaxis["axisLabel"]["formatter"] = echarts.label_js({m: theme.model_label(m) for m in models})
    option["xAxis"] = xaxis
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
                        "color": theme._rgba(color_map.get(m, "#9CA3AF"), 0.4),
                        "borderColor": color_map.get(m, "#9CA3AF"),
                    },
                }
                for m, stats, _ in rows
            ],
        }
    ]
    caption = "Box = p25–median–p75, whiskers p5–p95 (clipped above the tallest box)."
    if y_max is not None and n_above:
        caption += f" · {n_above} prompt(s) above ${y_max:,.2f}"
    return option, caption


_GRAIN_ADJ = {"Day": "Daily", "Week": "Weekly", "Month": "Monthly"}


def _cost_by_model_option(
    tokens: pd.DataFrame, primary: str, granularity: str
) -> dict[str, Any] | None:
    """Stacked bar of cost split by model, by ``granularity`` — the canonical emitter.

    One series per model (each its stable family color via ``model_color_map``);
    a click returns ``seriesName`` (the model), not the period category. Bucketing
    (Day / Week / Month) goes through ``data.to_period`` so it matches the app.
    """
    col = data.cost_col(primary)
    work = tokens.dropna(subset=["date", "model"])[["date", "model", col]].copy()
    work["period"] = data.to_period(work["date"], granularity)
    grouped = work.groupby(["period", "model"], as_index=False)[col].sum()
    grouped = grouped[grouped[col] > 0].sort_values("period")
    if grouped.empty:
        return None

    periods = sorted(grouped["period"].unique())
    labels = [data.period_label(p, granularity) for p in periods]
    models = theme.sort_models(grouped["model"].unique())
    color_map = theme.model_color_map(models)
    pivot = grouped.pivot_table(
        index="period", columns="model", values=col, aggfunc="sum", fill_value=0.0
    ).reindex(periods)
    series = [
        {
            "name": model,
            "type": "bar",
            "stack": "cost",
            "data": [round(float(v), 2) for v in pivot[model].tolist()],
            "itemStyle": {"color": color_map.get(model, "#9CA3AF")},
        }
        for model in models
    ]

    c = echarts.colors()
    labels_map = {m: theme.model_label(m) for m in models}
    option = echarts.base_option()
    option["title"] = {
        "text": f"{_GRAIN_ADJ[granularity]} cost by model ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"]["formatter"] = echarts.label_js(labels_map)
    option["grid"] = {"left": 56, "right": 24, "top": 48, "bottom": 64, "containLabel": True}
    option["tooltip"].update(
        {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "formatter": echarts.model_tooltip_js(labels_map, money=True),
        }
    )
    option["xAxis"] = echarts.category_axis(labels)
    option["yAxis"] = echarts.value_axis(money=True)
    option["series"] = series
    return option


def main() -> None:
    """Render the Models page."""
    from prompt_analytics.dashboard import echarts as ec

    st.title("Models")

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all)
    tokens = frames.get("tokens", pd.DataFrame())
    prompts = frames.get("prompts", pd.DataFrame())
    primary = data.primary_provider()

    if tokens.empty or "model" not in tokens.columns:
        st.info("No data for the current filters.")
        st.stop()

    st.caption("👆 Click a model on any chart to filter the whole dashboard to it.")

    # Split of usage and cost across models, side by side.
    left, right = st.columns(2)
    with left:
        tok_pie = _tokens_by_model_option(tokens)
        if tok_pie is None:
            st.info("No token volume for the current filters.")
        else:
            clicked = ec.render(tok_pie, key="models_tokens_pie", height="360px", click=True)
            ec.apply_click(clicked, filters.KEY_MODELS)
    with right:
        cost_pie = _cost_by_model_pie_option(tokens, primary)
        if cost_pie is None:
            st.info("No model cost for the current filters.")
        else:
            clicked = ec.render(cost_pie, key="models_cost_pie", height="360px", click=True)
            ec.apply_click(clicked, filters.KEY_MODELS)

    # Per-prompt cost distribution (median visible) by model.
    dist = _prompt_cost_distribution_option(prompts, primary)
    if dist is None:
        st.info("No per-prompt model data available.")
    else:
        dist_opt, dist_caption = dist
        clicked = ec.render(dist_opt, key="models_prompt_dist", height="400px", click=True)
        ec.apply_click(clicked, filters.KEY_MODELS)
        st.caption(dist_caption)

    # Cost split by model over time. Grain read from the toggle rendered *below*
    # the chart (session_state), default span-appropriate — consistent with the
    # Home and Overview charts.
    if "date" in tokens.columns:
        grain = st.session_state.get("models_granularity", data.auto_granularity(tokens["date"]))
        daily_opt = _cost_by_model_option(tokens, primary, grain)
    else:
        daily_opt = None
    if daily_opt is None:
        st.info("No dated cost for the current filters.")
    else:
        clicked = ec.render(
            daily_opt, key="models_daily", height="420px", click=True, click_field="seriesName"
        )
        ec.apply_click(clicked, filters.KEY_MODELS)
        filters.granularity_control(tokens["date"], key="models_granularity")

    st.caption(
        "Costs use the primary provider's grid. For the subscription-vs-usage "
        "comparison (Claude plans, GitHub Copilot), see the **Quotas** page."
    )


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
