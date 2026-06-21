"""Prompt Explorer: drill from session -> prompt with detailed tables.

The single place to inspect raw detail, so the analytical pages can stay light.
It respects the **global cross-filters** (a project / model / date / category
clicked on any chart carries through here -- that is the "drill-through on the
current selection"), and adds a local **session -> prompt** drill via
``session_state``: pick a session to see its prompts. Charts elsewhere deep-link
here by setting ``drill_session`` (e.g. the Usage top-10 table, the Sessions
treemap), which preselects that session straight away.

Mostly table-first (``st.dataframe`` row selection for the drill); the one chart
is the per-session **cumulative cost** timeline at the prompt level. Because it
renders ECharts, it is excluded from the headless AppTest enumeration (like the
other chart pages) and ``main()`` runs only under a real server.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics.dashboard import data, echarts, filters, tables, theme

DRILL_SESSION = "drill_session"
# The File Explorer reads these to preselect a file's drill (kept as literals so
# this page need not import the numbered page module, which renders on import).
# ``DRILL_FILE_PROJECT`` disambiguates which repo's copy of the path to drill into.
DRILL_FILE = "drill_file"
DRILL_FILE_PROJECT = "drill_file_project"
_FILE_EXPLORER_PAGE = "pages/12_file_explorer.py"

# The two row-selection tables. Their keys are popped when leaving a drill so the
# sticky ``st.dataframe`` selection can't re-apply the just-cleared drill on the
# next rerun (these literals also appear in ``filters._DRILL_KEYS`` so the global
# Reset clears them too).
_KEY_SESSIONS_TABLE = "explorer_sessions"
_KEY_PROMPTS_TABLE = "explorer_prompts"


def _files_edited_by(prompt_id: str, ds: Any | None = None) -> list[str]:
    """Project-relative paths the given prompt edited (for the deep-link out).

    ``ds`` defaults to the active dataset; it is injectable so the helper can be
    unit-tested without a data directory.
    """
    if not prompt_id:
        return []
    if ds is None:
        ds = data.load_dataset()
    seen: dict[str, None] = {}
    for row in ds.output_files:
        if str(row.get("prompt_id") or "") != prompt_id:
            continue
        path = str(row.get("path") or "")
        if path and path != "-":
            seen.setdefault(path, None)
    return list(seen)


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
    """Render the Prompt Explorer page."""
    st.title("Prompt Explorer")

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
        "(click a chart anywhere, then come here); pick a session to see its prompts."
    )

    # Resolve session focus: a focused session = deep-linked (Sessions treemap /
    # Usage top-10) or selected below; the table then narrows to that one session.
    by_session = _by_session(prompts, tokens, sessions, cost)
    drill = st.session_state.get(DRILL_SESSION)
    focused = (
        drill
        if (drill and not by_session.empty and drill in set(by_session["session_id"]))
        else None
    )

    # --- Session level -----------------------------------------------------
    st.subheader("Sessions")
    if by_session.empty:
        st.info("No sessions for the current selection.")
        return
    # The focused session (resolved above) narrows the table to that one session,
    # so the page is clearly filtered to it; "← All sessions" returns to the list.
    if focused:
        fcol = st.columns([5, 1])
        fcol[0].caption(f"🔎 Focused on session **{focused[:8]}…** — its prompts are below.")
        if fcol[1].button("← All sessions", width="stretch", key="clear_drill_session"):
            # Pop the table selections too: otherwise the sticky row-selection
            # re-applies the same drill on the next rerun (the "← back doesn't
            # return to the list" bug) -- it would re-focus the top session.
            for k in (DRILL_SESSION, _KEY_SESSIONS_TABLE, _KEY_PROMPTS_TABLE):
                st.session_state.pop(k, None)
            st.rerun()
    table_src = (
        by_session[by_session["session_id"] == focused].reset_index(drop=True)
        if focused
        else by_session
    )
    cols = [c for c in ["session_id", "project", "model", "day", "prompts", cost] if c in table_src]
    view = table_src[cols].rename(
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
        tables.bar_table(view, count_cols=("Prompts",), cost_cols=("Cost (USD)",)),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=_KEY_SESSIONS_TABLE,
    )
    rows = list(event.get("selection", {}).get("rows", []))
    # Bound-check: a sticky selection index can outlive the table when it shrinks to
    # the focused session on the next rerun (else iloc raises out-of-bounds).
    if rows and rows[0] < len(table_src):
        picked = str(table_src.iloc[rows[0]]["session_id"])
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
        c
        for c in ["prompt_index", "model", "category", "char_count", cost, "prompt_preview"]
        if c in sub.columns
    ]
    if sub.empty or not pcols:
        st.info("No prompts for this session.")
        return
    detail = sub[pcols].rename(
        columns={
            "prompt_index": "#",
            "model": "Model",
            "category": "Category",
            "char_count": "Chars",
            cost: "Cost (USD)",
            "prompt_preview": "Prompt",
        }
    )
    if "Model" in detail.columns:
        detail["Model"] = detail["Model"].map(lambda m: theme.model_label(m) if pd.notna(m) else m)
    event = st.dataframe(
        tables.bar_table(detail, count_cols=("Chars",), cost_cols=("Cost (USD)",)),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=_KEY_PROMPTS_TABLE,
    )
    if cost in sub.columns:
        st.caption(
            f"{len(sub)} prompts · ${sub[cost].fillna(0).sum():,.2f} total · select a row above to read the full prompt"
        )

    # Full prompt text on demand: the table shows only the truncated preview, so
    # a selected row reveals the complete prompt (from prompts_text.csv if present).
    rows = list(event.get("selection", {}).get("rows", []))
    if rows and rows[0] < len(sub):
        row = sub.iloc[rows[0]]
        texts = data.load_prompt_texts()
        full = texts.get(str(row.get("prompt_id", "")), "") or str(row.get("prompt_preview", ""))
        with st.expander("Full prompt", expanded=True):
            if full.strip():
                st.markdown(f"> {full}".replace("\n", "\n> "))
            else:
                st.info("No prompt text available (extracted with --no-text, or text disabled).")

        # Deep-link out: the files this prompt edited, each jumping to the File
        # Explorer with that file's drill preselected (same principle as Explore →).
        edited_files = _files_edited_by(str(row.get("prompt_id", "")))
        if edited_files:
            st.caption("Files this prompt edited — open one in the **File Explorer**:")
            for i, path in enumerate(edited_files):
                if st.button(f"📄 {path} →", key=f"explore_file_{i}", width="stretch"):
                    st.session_state[DRILL_FILE] = path
                    st.session_state[DRILL_FILE_PROJECT] = str(row.get("project") or "")
                    # Clear any stale row-selection on the target table, else its
                    # sticky pick would override this deep-link on arrival ("fe_files"
                    # is the File Explorer's table key).
                    st.session_state.pop("fe_files", None)
                    st.switch_page(_FILE_EXPLORER_PAGE)


# Render only under a real Streamlit server: the timeline is ECharts, which can't
# register under a bare import / AppTest. Excluded from the headless enumeration.
if runtime.exists():
    main()
