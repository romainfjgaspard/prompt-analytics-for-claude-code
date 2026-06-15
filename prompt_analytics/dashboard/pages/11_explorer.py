"""Explorer: drill from day -> session -> prompt with detailed tables.

The single place to inspect raw detail, so the analytical pages can stay light.
It respects the **global cross-filters** (a project / model / date / category
clicked on any chart carries through here -- that is the "drill-through on the
current selection"), and adds a local **day -> session -> prompt** drill via
``session_state``: pick a day to narrow the session list, pick a session to see
its prompts. Charts elsewhere deep-link here by setting ``drill_session`` (e.g.
the Usage top-10 table, the Sessions treemap).

Mostly table-first (``st.dataframe`` row selection for the drill); the one chart
is the per-session **cumulative cost** timeline at the prompt level. Because it
renders ECharts, it is excluded from the headless AppTest enumeration (like the
other chart pages) and ``main()`` runs only under a real server.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics.dashboard import data, echarts, filters, theme

DRILL_DATE = "drill_date"
DRILL_SESSION = "drill_session"


def _title(text: str) -> dict[str, Any]:
    """A left-aligned chart title in the section's type hierarchy."""
    c = echarts.colors()
    return {
        "text": text,
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }


def _session_timeline_option(sub: pd.DataFrame, cost: str, primary: str) -> dict[str, Any] | None:
    """Cumulative cost prompt by prompt for one session (7.3), annotated.

    Marks the points that tell the story: the opening prompt, the costliest
    prompt, and the running total at the end.
    """
    if not {"prompt_index", cost} <= set(sub.columns):
        return None
    work = sub.dropna(subset=["prompt_index"]).copy()
    work["prompt_index"] = pd.to_numeric(work["prompt_index"], errors="coerce")
    work = work.dropna(subset=["prompt_index"]).sort_values("prompt_index")
    if work.empty:
        return None
    work[cost] = work[cost].fillna(0.0)
    work["cumulative"] = work[cost].cumsum()

    line_data = [
        {"value": [int(pi), round(float(cum), 2)], "cost": round(float(cst), 2)}
        for pi, cum, cst in zip(work["prompt_index"], work["cumulative"], work[cost], strict=True)
    ]

    c = echarts.colors()
    accent = theme.PALETTE[0]
    first = work.iloc[0]
    costliest = work.loc[work[cost].idxmax()]
    last = work.iloc[-1]

    annotations: list[dict[str, Any]] = [
        {
            "value": [int(first["prompt_index"]), round(float(first["cumulative"]), 2)],
            "symbolSize": 8,
            "itemStyle": {"color": accent},
            "label": {
                "show": True,
                "position": "top",
                "color": c["muted"],
                "fontSize": 11,
                "formatter": f"opens at ${first[cost]:,.2f}",
            },
        }
    ]
    if costliest["prompt_index"] != first["prompt_index"]:
        annotations.append(
            {
                "value": [int(costliest["prompt_index"]), round(float(costliest["cumulative"]), 2)],
                "symbolSize": 8,
                "itemStyle": {"color": accent},
                "label": {
                    "show": True,
                    "position": "top",
                    "color": c["muted"],
                    "fontSize": 11,
                    "formatter": f"costliest: p{int(costliest['prompt_index'])} (${costliest[cost]:,.2f})",
                },
            }
        )
    annotations.append(
        {
            "value": [int(last["prompt_index"]), round(float(last["cumulative"]), 2)],
            "symbolSize": 8,
            "itemStyle": {"color": accent},
            "label": {
                "show": True,
                "position": "right",
                "color": accent,
                "fontWeight": 600,
                "fontSize": 12,
                "formatter": f"total ${last['cumulative']:,.2f}",
            },
        }
    )

    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = _title(f"Cumulative cost, prompt by prompt ({primary})")
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "trigger": "item",
        "formatter": echarts.js(
            "function(p){var d=p.data;if(d.cost===undefined){return '';}"
            "return 'Prompt '+d.value[0]+'<br/>Cumulative: $'+Number(d.value[1]).toFixed(2)"
            "+'<br/>This prompt: $'+Number(d.cost).toFixed(2);}"
        ),
    }
    option["xAxis"] = {
        "type": "value",
        "name": "Prompt index in session",
        "nameLocation": "middle",
        "nameGap": 30,
        "nameTextStyle": {"color": c["muted"]},
        "min": 1,  # prompt indices start at 1, not 0
        "minInterval": 1,
        "axisLabel": {"color": c["text"]},
        "axisLine": {"lineStyle": {"color": c["axis"]}},
        "splitLine": {"lineStyle": {"color": c["grid"]}},
    }
    # nameLocation "middle" keeps "Cumulative cost" off the top-left, where it
    # overlapped the chart title.
    yaxis = echarts.value_axis(money=True, name="Cumulative cost")
    yaxis["nameLocation"] = "middle"
    yaxis["nameGap"] = 56
    option["yAxis"] = yaxis
    option["series"] = [
        {
            "type": "line",
            "data": line_data,
            "showSymbol": True,
            "symbolSize": 6,
            "lineStyle": {"color": accent, "width": 2},
            "itemStyle": {"color": accent},
            "areaStyle": {"color": theme._rgba(accent, 0.10)},
        },
        {"type": "scatter", "data": annotations, "z": 6, "tooltip": {"show": False}},
    ]
    return option


def _session_first_date(prompts: pd.DataFrame) -> pd.DataFrame:
    """``[session_id, day]`` using each session's earliest prompt date."""
    if prompts.empty or not {"session_id", "date"} <= set(prompts.columns):
        return pd.DataFrame(columns=["session_id", "day"])
    dated = prompts.dropna(subset=["session_id", "date"])
    first = dated.groupby("session_id", as_index=False)[["date"]].min()
    first["day"] = pd.to_datetime(first["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out: pd.DataFrame = first[["session_id", "day"]]
    return out


def _by_day(prompts: pd.DataFrame, tokens: pd.DataFrame, cost: str) -> pd.DataFrame:
    """Per-day rollup: sessions, prompts and cost (newest first)."""
    if prompts.empty or "date" not in prompts.columns:
        return pd.DataFrame()
    p = prompts.dropna(subset=["date"]).copy()
    p["day"] = pd.to_datetime(p["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    agg = p.groupby("day").agg(sessions=("session_id", "nunique"), prompts=("prompt_id", "nunique"))
    if not tokens.empty and "date" in tokens.columns and cost in tokens.columns:
        t = tokens.dropna(subset=["date"]).copy()
        t["day"] = pd.to_datetime(t["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        agg["cost"] = t.groupby("day")[cost].sum()
    agg["cost"] = agg.get("cost", pd.Series(dtype=float)).fillna(0.0)
    out: pd.DataFrame = agg.reset_index().sort_values("day", ascending=False).reset_index(drop=True)
    return out


def _by_session(
    prompts: pd.DataFrame, tokens: pd.DataFrame, sessions: pd.DataFrame, cost: str
) -> pd.DataFrame:
    """Per-session rollup: project, dominant model, day, prompts and cost."""
    if prompts.empty or "session_id" not in prompts.columns:
        return pd.DataFrame()
    counts = prompts.groupby("session_id").size().reset_index(name="prompts")
    result = counts.merge(_session_first_date(prompts), on="session_id", how="left")
    result = result.merge(data.dominant_model_per_session(prompts), on="session_id", how="left")
    if not tokens.empty and "session_id" in tokens.columns and cost in tokens.columns:
        per_session = (
            tokens.dropna(subset=["session_id"]).groupby("session_id", as_index=False)[cost].sum()
        )
        result = result.merge(per_session, on="session_id", how="left")
    if not sessions.empty and {"session_id", "project"} <= set(sessions.columns):
        result = result.merge(sessions[["session_id", "project"]], on="session_id", how="left")
    if cost not in result.columns:
        result[cost] = 0.0
    result[cost] = result[cost].fillna(0.0)
    out: pd.DataFrame = result.sort_values(cost, ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    """Render the Explorer page."""
    st.title("Explorer")

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all, explore_link=False)
    prompts = frames.get("prompts", pd.DataFrame())
    tokens = frames.get("tokens", pd.DataFrame())
    sessions = frames.get("sessions", pd.DataFrame())
    primary = data.primary_provider()
    cost = data.cost_col(primary)

    if prompts.empty or "session_id" not in prompts.columns:
        st.info("No data for the current filters.")
        st.stop()

    st.caption(
        "The detail drill. It already reflects the dashboard's active filters "
        "(click a chart anywhere, then come here); pick a day to narrow the "
        "sessions, then a session to see its prompts."
    )

    # --- Day level ---------------------------------------------------------
    day_filter = st.session_state.get(DRILL_DATE)
    by_day = _by_day(prompts, tokens, cost)
    st.subheader("By day")
    if by_day.empty:
        st.info("No dated activity.")
    else:
        view = by_day.rename(
            columns={
                "day": "Day",
                "sessions": "Sessions",
                "prompts": "Prompts",
                "cost": "Cost (USD)",
            }
        )
        event = st.dataframe(
            view.style.format({"Cost (USD)": "${:,.2f}"}),
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="explorer_days",
        )
        rows = list(event.get("selection", {}).get("rows", []))
        if rows:
            picked = str(by_day.iloc[rows[0]]["day"])
            if picked != day_filter:
                st.session_state[DRILL_DATE] = picked
                st.session_state.pop(DRILL_SESSION, None)  # day changed -> reset session
                st.rerun()
        if day_filter:
            left, right = st.columns([5, 1])
            left.caption(f"Filtered to **{day_filter}** — sessions below are limited to that day.")
            if right.button("Clear day", width="stretch"):
                st.session_state.pop(DRILL_DATE, None)
                st.rerun()

    # --- Session level -----------------------------------------------------
    by_session = _by_session(prompts, tokens, sessions, cost)
    if day_filter and not by_session.empty and "day" in by_session.columns:
        by_session = by_session[by_session["day"] == day_filter].reset_index(drop=True)
    st.subheader("Sessions" + (f" on {day_filter}" if day_filter else ""))
    if by_session.empty:
        st.info("No sessions for the current selection.")
        return
    cols = [
        c for c in ["session_id", "project", "model", "day", "prompts", cost] if c in by_session
    ]
    view = by_session[cols].rename(
        columns={
            "session_id": "Session",
            "project": "Project",
            "model": "Model",
            "day": "Day",
            "prompts": "Prompts",
            cost: "Cost (USD)",
        }
    )
    if "Model" in view.columns:
        view["Model"] = view["Model"].map(lambda m: theme.model_label(m) if pd.notna(m) else m)
    event = st.dataframe(
        view.style.format({"Cost (USD)": "${:,.2f}"}),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="explorer_sessions",
    )
    rows = list(event.get("selection", {}).get("rows", []))
    if rows:
        picked = str(by_session.iloc[rows[0]]["session_id"])
        if picked != st.session_state.get(DRILL_SESSION):
            st.session_state[DRILL_SESSION] = picked
            st.rerun()

    # --- Prompt level ------------------------------------------------------
    selected = st.session_state.get(DRILL_SESSION)
    valid = set(by_session["session_id"])
    if selected not in valid:
        st.info("Select a session above to see its prompts.")
        return
    st.subheader(f"Prompts in session {selected}")
    sub = prompts[prompts["session_id"] == selected].copy()
    if "prompt_index" in sub.columns:
        sub = sub.sort_values("prompt_index")
    timeline = _session_timeline_option(sub, cost, primary)
    if timeline is not None:
        echarts.render(timeline, key="explorer_timeline", height="360px")
    pcols = [
        c for c in ["prompt_index", "model", "category", cost, "prompt_preview"] if c in sub.columns
    ]
    if sub.empty or not pcols:
        st.info("No prompts for this session.")
        return
    detail = sub[pcols].rename(
        columns={
            "prompt_index": "#",
            "model": "Model",
            "category": "Category",
            cost: "Cost (USD)",
            "prompt_preview": "Prompt",
        }
    )
    if "Model" in detail.columns:
        detail["Model"] = detail["Model"].map(lambda m: theme.model_label(m) if pd.notna(m) else m)
    # Annotated to match Styler.format's formatter type exactly (dict value type
    # is invariant, so a bare dict[str, str] is rejected by pandas-stubs).
    detail_fmt: dict[Any, str | Callable[[object], str] | None] = (
        {"Cost (USD)": "${:,.2f}"} if cost in sub.columns else {}
    )
    event = st.dataframe(
        detail.style.format(detail_fmt),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="explorer_prompts",
    )
    if cost in sub.columns:
        st.caption(
            f"{len(sub)} prompts · ${sub[cost].fillna(0).sum():,.2f} total · select a row above to read the full prompt"
        )

    # Full prompt text on demand: the table shows only the truncated preview, so
    # a selected row reveals the complete prompt (from prompts_text.csv if present).
    rows = list(event.get("selection", {}).get("rows", []))
    if rows:
        row = sub.iloc[rows[0]]
        texts = data.load_prompt_texts()
        full = texts.get(str(row.get("prompt_id", "")), "") or str(row.get("prompt_preview", ""))
        with st.expander("Full prompt", expanded=True):
            if full.strip():
                st.markdown(f"> {full}".replace("\n", "\n> "))
            else:
                st.info("No prompt text available (extracted with --no-text, or text disabled).")


# Render only under a real Streamlit server: the timeline is ECharts, which can't
# register under a bare import / AppTest. Excluded from the headless enumeration.
if runtime.exists():
    main()
