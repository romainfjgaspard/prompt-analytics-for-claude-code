"""Streamlit dashboard landing page for prompt-analytics-for-claude-code.

The landing stays above the fold: one compact "where the bill goes" bar (cost
split by token type), then the spend-trend KPIs with sparklines and a delta vs
the prior 7 days (7.2, the burn-rate analysis), then the daily cost stacked by
model. The two hero *numbers* that used to head this page moved to where their
supporting detail already lives — the context-rent share to the Usage page
(above its cost-by-token-type trend), and the depth-cost story is the Session
depth page's own headline. The subscription-vs-usage cost comparison (Claude
plans, GitHub Copilot) lives on the Quotas page.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics.dashboard import data as data_mod
from prompt_analytics.dashboard import filters as filters_mod
from prompt_analytics.dashboard import theme

# NB: ``echarts`` is imported lazily inside the render helpers, never at module
# top. Importing ``streamlit_echarts`` registers a frontend component, which
# raises "must be declared in pyproject.toml" under a bare import / AppTest
# (empty registry); it only succeeds under a real ``streamlit run``. Keeping the
# import local lets the pure calc helpers (and their unit tests) import this
# module without a running server.

NO_DATA_MESSAGE = "No data — run `prompt-analytics extract`."

# Back-compat alias: the date-bounds helper now lives in ``filters`` (shared by
# every page's sidebar), but tests and older imports reference it here.
_available_date_bounds = filters_mod.available_date_bounds

# Display order of the cost-split bar: rent components first, then generation.
_SPLIT_ORDER = ("cache_read", "cache_write_5m", "cache_write_1h", "output", "input")


# ---------------------------------------------------------------------------
# "Where the bill goes" cost-split bar (7.1). The two hero *numbers* moved to
# their thematic pages — the context-rent share to Usage (above its cost-by-
# token-type trend), the depth-cost story is the Session-depth page's own
# headline — so the landing pairs the spend-trend KPIs with this compact donut,
# then the cost-over-time chart, all above the fold.
# ---------------------------------------------------------------------------


def _cost_split_donut(tokens: pd.DataFrame, primary: str) -> dict[str, Any] | None:
    """A donut of where the bill goes, by token type, sitting beside the KPIs.

    Compact enough to share a row with the KPI cards (so the cost-over-time
    chart stays above the fold). One slice per token type in rent-first order,
    brand token-type colors preserved, percentage labels on the ring.
    """
    col = data_mod.cost_col(primary)
    if tokens.empty or col not in tokens.columns:
        return None
    by_type = tokens.groupby("token_type")[col].sum()
    points: list[dict[str, Any]] = []
    for token_type in _SPLIT_ORDER:
        value = float(by_type.get(token_type, 0.0))
        if value <= 0:
            continue
        label = str(tokens.loc[tokens["token_type"] == token_type, "token_type_label"].iloc[0])
        points.append(
            {
                "name": label,
                "value": round(value, 2),
                "itemStyle": {"color": theme.TOKEN_TYPE_COLORS.get(token_type, "#9CA3AF")},
            }
        )
    if not points:
        return None

    from prompt_analytics.dashboard import echarts

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": f"Where the bill goes ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["tooltip"] = {
        "trigger": "item",
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "formatter": "{b}: ${c} ({d}%)",
    }
    option["series"] = [
        {
            "type": "pie",
            "radius": ["48%", "74%"],
            "center": ["50%", "55%"],
            "avoidLabelOverlap": True,
            "label": {"show": True, "formatter": "{d}%", "color": c["text"], "fontSize": 11},
            "labelLine": {"length": 6, "length2": 6},
            "data": points,
        }
    ]
    return option


# ---------------------------------------------------------------------------
# Spend-trend KPIs with sparklines and deltas (7.2).
# ---------------------------------------------------------------------------


def _daily_series(frames: dict[str, pd.DataFrame], primary: str) -> pd.DataFrame:
    """Daily sessions / prompts / tokens / cost over the full observed range.

    Missing days are kept as zeros so the sparklines show the real rhythm
    (gaps included) instead of joining active days together.
    """
    prompts = frames.get("prompts", pd.DataFrame())
    tokens = frames.get("tokens", pd.DataFrame())
    col = data_mod.cost_col(primary)

    parts: list[pd.Series] = []
    if not prompts.empty and "date" in prompts.columns:
        dated = prompts.dropna(subset=["date"])
        parts.append(dated.groupby("date")["prompt_id"].nunique().rename("prompts"))
        if "session_id" in dated.columns:
            parts.append(dated.groupby("date")["session_id"].nunique().rename("sessions"))
    if not tokens.empty and "date" in tokens.columns and col in tokens.columns:
        dated_tokens = tokens.dropna(subset=["date"])
        volume = dated_tokens[dated_tokens["token_type"] != "server_tool_use"]
        parts.append(volume.groupby("date")["token_count"].sum().rename("tokens"))
        parts.append(dated_tokens.groupby("date")[col].sum().rename("cost"))
    if not parts:
        return pd.DataFrame()

    daily = pd.concat(parts, axis=1)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    result: pd.DataFrame = daily.reindex(full_range).fillna(0.0)
    return result


def _delta_vs_prior_week(series: pd.Series) -> str | None:
    """``+42% vs prior 7 days`` for a date-indexed daily series, or None."""
    if series.empty:
        return None
    last = series.index.max()
    age_days = (last - series.index).days
    last7 = float(series[age_days < 7].sum())
    prior7 = float(series[(age_days >= 7) & (age_days < 14)].sum())
    if prior7 == 0:
        return None
    return f"{100 * (last7 - prior7) / prior7:+.0f}% vs prior 7 days"


def _render_kpis(frames: dict[str, pd.DataFrame], primary: str) -> None:
    """KPI cards with sparklines and week-over-week deltas (7.2)."""
    from prompt_analytics.dashboard import echarts

    sessions = frames.get("sessions", pd.DataFrame())
    prompts = frames.get("prompts", pd.DataFrame())
    tokens = frames.get("tokens", pd.DataFrame())

    total_sessions = (
        int(sessions["session_id"].nunique()) if "session_id" in sessions.columns else 0
    )
    total_prompts = int(prompts["prompt_id"].nunique()) if "prompt_id" in prompts.columns else 0
    if {"token_type", "token_count"} <= set(tokens.columns):
        total_tokens = int(
            tokens.loc[tokens["token_type"] != "server_tool_use", "token_count"].sum()
        )
    else:
        total_tokens = 0
    col = data_mod.cost_col(primary)
    total_cost = float(tokens[col].sum()) if col in tokens.columns else 0.0

    daily = _daily_series(frames, primary)

    tokens_m = total_tokens / 1_000_000
    # Tokens are in the hundreds of millions: show the unit, not a 9-digit count.
    tokens_str = f"{tokens_m:,.0f}M" if tokens_m >= 10 else f"{tokens_m:,.1f}M"

    # (label, value, daily-series column, delta color, sparkline color)
    cards: list[tuple[str, str, str, Literal["off", "inverse"], str]] = [
        ("Total sessions", f"{total_sessions:,}", "sessions", "off", theme.PALETTE[5]),
        ("Total prompts", f"{total_prompts:,}", "prompts", "off", theme.PALETTE[1]),
        ("Total tokens", tokens_str, "tokens", "off", theme.PALETTE[2]),
        # More spend than last week reads as a warning (inverse: down = green).
        (f"Cost ({primary})", f"${total_cost:,.2f}", "cost", "inverse", theme.PALETTE[0]),
    ]
    # 2x2 grid so the cards sit in the left column beside the donut and stay
    # readable (4-across in a half-width column wraps the labels).
    grid = [st.columns(2), st.columns(2)]
    for i, (label, value, series_name, delta_color, color) in enumerate(cards):
        series = daily[series_name] if series_name in daily.columns else pd.Series(dtype=float)
        delta = _delta_vs_prior_week(series)
        with grid[i // 2][i % 2]:
            st.metric(label, value, delta=delta, delta_color=delta_color if delta else "off")
            if len(series) > 1:
                values = [float(v) for v in series.to_numpy()]
                echarts.render(
                    echarts.sparkline_option(values, color),
                    key=f"spark_{series_name}",
                    height="56px",
                )


_GRAIN_ADJ = {"Day": "Daily", "Week": "Weekly", "Month": "Monthly"}


def _cost_by_model_option(
    tokens: pd.DataFrame, primary: str, granularity: str
) -> dict[str, Any] | None:
    """Stacked bar of the primary provider's cost, split by model, by ``granularity``.

    A cross-filter *emitter*: clicking a model's segment narrows the whole
    dashboard to that model (``KEY_MODELS``); the click returns ``seriesName``
    (the model) rather than the category (the period). One series per model, each
    pinned to its stable family color via ``theme.model_color_map``. Bucketing
    (Day / Week / Month) goes through :func:`data.to_period` so the app and the
    Models page group identically.
    """
    from prompt_analytics.dashboard import echarts

    col = data_mod.cost_col(primary)
    work = tokens.dropna(subset=["date"])[["date", "model", col]].copy()
    work["period"] = data_mod.to_period(work["date"], granularity)
    grouped = work.groupby(["period", "model"], as_index=False)[col].sum()
    grouped = grouped[grouped[col] > 0].sort_values("period")
    if grouped.empty:
        return None

    periods = sorted(grouped["period"].unique())
    labels = [data_mod.period_label(p, granularity) for p in periods]
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
    # Friendly model names in legend + tooltip; series.name stays the raw id so
    # the click cross-filter (seriesName -> KEY_MODELS) keeps matching the data.
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


def _home() -> None:
    """Render the landing page: KPIs + where-the-bill-goes donut, then cost over time."""
    from prompt_analytics.dashboard import echarts

    st.title("Prompt Analytics for Claude Code")

    frames = data_mod.load_all()
    primary = data_mod.primary_provider()

    prompts = frames.get("prompts", pd.DataFrame())
    tokens = frames.get("tokens", pd.DataFrame())
    if prompts.empty and tokens.empty:
        st.info(NO_DATA_MESSAGE)
        return

    filters_mod.render_sidebar(frames)
    filtered = filters_mod.apply_filters(frames)
    filters_mod.render_active_filter_badge(frames)

    if filtered["prompts"].empty and filtered["tokens"].empty:
        st.info("No data matches the current filters.")
        return

    # KPI cards (left) beside the where-the-bill-goes donut (right) in one row, so
    # the cost-over-time chart below stays above the fold.
    left, right = st.columns([3, 2])
    with left:
        _render_kpis(filtered, primary)
    with right:
        donut = _cost_split_donut(filtered["tokens"], primary)
        if donut is not None:
            echarts.render(donut, key="cost_split", height="300px")

    # No section header here: the chart carries its own title ("… cost by model").
    tok = filtered["tokens"]
    if tok.empty or "date" not in tok.columns:
        st.info("No cost data to plot.")
    else:
        # The Group-by control sits *below* the chart; read its current value (or
        # the span-based default) up here so the chart can build before it renders.
        grain_key = "app_granularity"
        grain = st.session_state.get(grain_key, data_mod.auto_granularity(tok["date"]))
        option = _cost_by_model_option(tok, primary, grain)
        if option is None:
            st.info("No cost data to plot.")
        else:
            clicked = echarts.render(
                option, key="daily_by_model", height="420px", click=True, click_field="seriesName"
            )
            echarts.apply_click(clicked, filters_mod.KEY_MODELS)
        filters_mod.granularity_control(tok["date"], key=grain_key)
        st.caption(
            "👆 Click a model to filter the dashboard. "
            "Costs are priced on the primary provider's grid; the "
            "subscription-vs-usage comparison (incl. GitHub Copilot) is on the Quotas page."
        )


def _run() -> None:
    """Build and run the dashboard navigation (real Streamlit server only).

    Moving off Streamlit's automatic ``pages/`` discovery to ``st.navigation``
    lets every tab carry an explicit, capitalized title: the auto-nav derives the
    label from the filename, so the entry script showed "app" and the pages showed
    lowercase names ("overview", "session_depth"). The home content lives in
    :func:`_home`; every other tab is its page file. ``set_page_config`` is called
    once here, so the page files no longer call it themselves. The demo banner is
    rendered at the bottom of the sidebar (in :func:`filters.render_sidebar`), not
    here, so it sits below the filters instead of in each page's main column.
    """
    st.set_page_config(layout="wide", page_title="Prompt Analytics for Claude Code")
    # Trim Streamlit's large default top padding (~6rem) so the page title sits
    # close to the top toolbar instead of floating below a gap. Applied on every
    # page (entry script runs on each load); `.block-container` is the main area.
    st.markdown(
        "<style>.block-container{padding-top:2.5rem;}</style>",
        unsafe_allow_html=True,
    )
    pages = [
        st.Page(_home, title="Home", icon="🏠", default=True),
        st.Page("pages/1_overview.py", title="Usage", icon="📊"),
        st.Page("pages/2_models.py", title="Models", icon="🧠"),
        st.Page("pages/5_sessions.py", title="Sessions", icon="🗂️"),
        st.Page("pages/4_session_depth.py", title="Session depth", icon="📐"),
        st.Page("pages/3_prompts.py", title="Prompts", icon="💬"),
        # Composition narrates the "where the cost goes, by content" spine; it
        # respects the global filters, so it sits in the filter-driven block next
        # to Prompts (the input side of the same story).
        st.Page("pages/8_composition.py", title="Composition", icon="🧩"),
        # Explorer respects the global filters, so it closes the filter-driven
        # analytics block; the pages that ignore the sidebar filters (Optimize is
        # request-grain, Quotas and How it works have none) come after.
        st.Page("pages/11_explorer.py", title="Explorer", icon="🔎"),
        st.Page("pages/6_optimize.py", title="Optimize", icon="✨"),
        st.Page("pages/7_quotas.py", title="Quotas", icon="📏"),
        st.Page("pages/10_how_it_works.py", title="How it works", icon="❓"),
    ]
    st.navigation(pages).run()


# Render only under a real Streamlit server. A bare ``import app`` (the unit tests
# for the pure calc helpers above) must not execute the render path: ECharts'
# frontend registration raises outside a running server. ``streamlit run`` sets
# runtime.exists() True; a bare import leaves it False.
if runtime.exists():
    _run()
