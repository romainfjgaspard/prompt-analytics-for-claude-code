"""Sessions page: where the money goes by project/session, plus a drill-down.

Migrated to Apache ECharts (``docs/MIGRATION-ECHARTS.md``). Emitters (§4):

* the **project pareto** is a cross-filter emitter -- clicking a bar narrows the
  whole dashboard to that project (``filters.KEY_PROJECTS``);
* the **treemap** is a *drill* trigger, not a filter -- clicking a session tile
  opens that session in the Prompt Explorer (``st.session_state['drill_session']``);
* the **per-session cost box plot** emits the model dimension
  (``filters.KEY_MODELS``), consistent with the Models page;
* the **prompts-per-session bar** emits a *new* global dimension
  (``filters.XF_PROMPT_COUNT``) -- clicking the "1" bar narrows the whole
  dashboard to one-prompt sessions, and so on.

Per-session / per-prompt detail lives on the Prompt Explorer page.

⚠️ Color stability: one ``project_color_map`` is built **once** from the
*unfiltered* project universe (``frames_all``) and shared by the pareto and
treemap, so a project keeps the same hue across both even though each filters a
different subset (see ``theme.project_color_map`` docstring). Zero Plotly remains.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics.dashboard import data, echarts, filters, theme

# Sticky-guard markers: the last value actually applied per interactive chart,
# so the component's sticky value never re-applies after a rerun / Reset.
_XF_TREEMAP_APPLIED = "_xf_treemap_applied"
# The drill-down selectbox key (also set by the Overview top-10 table).
DRILL_KEY = "drill_session"

_SYNTHETIC = frozenset({"(session overhead)", "(unknown)", ""})


def _per_session_cost(tokens: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Sum the primary provider's cost per session_id."""
    col = data.cost_col(primary)
    # ``[[col]]`` (not ``[col]``): with as_index=False both return a DataFrame at
    # runtime, but pandas-stubs only types the list form as DataFrame.
    result: pd.DataFrame = (
        tokens.dropna(subset=["session_id"]).groupby("session_id", as_index=False)[[col]].sum()
    )
    return result


def _title(text: str) -> dict[str, Any]:
    """A left-aligned chart title in the section's type hierarchy."""
    c = echarts.colors()
    return {
        "text": text,
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }


def _project_pareto_option(
    tokens: pd.DataFrame, primary: str, color_map: dict[str, str]
) -> tuple[dict[str, Any], int] | None:
    """Horizontal pareto of cost per project (biggest on top), cumulative % labels.

    A click returns the project ``name`` -> ``KEY_PROJECTS``. Returns the option
    plus a suggested pixel height (one row per project, so labels never crowd).
    """
    col = data.cost_col(primary)
    if "project" not in tokens.columns:
        return None
    work = tokens.copy()
    work["project"] = work["project"].fillna("(session overhead)").replace("", "(unknown)")
    agg = work.groupby("project", as_index=False)[[col]].sum().sort_values(col, ascending=False)
    agg = agg[agg[col] > 0]
    total = float(agg[col].sum())
    if not total:
        return None
    agg["share"] = 100 * agg[col] / total
    agg["cumulative"] = agg["share"].cumsum()

    c = echarts.colors()
    projects = agg["project"].tolist()
    data_items = [
        {
            "value": round(float(cost_v), 2),
            "itemStyle": {"color": color_map.get(proj, "#9CA3AF"), "borderRadius": [0, 4, 4, 0]},
            "label": {
                "show": True,
                "position": "right",
                "color": c["muted"],
                "formatter": f"{share:.0f}%  (cum. {cum:.0f}%)",
            },
        }
        for proj, cost_v, share, cum in zip(
            agg["project"], agg[col], agg["share"], agg["cumulative"], strict=True
        )
    ]

    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["grid"] = {"left": 8, "right": 88, "top": 48, "bottom": 24, "containLabel": True}
    option["title"] = _title(f"Cost by project ({primary})")
    option["tooltip"].update(
        {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "valueFormatter": echarts.js("function(v){return '$'+Number(v).toFixed(2);}"),
        }
    )
    # value on x, projects on y (inverse so the biggest, listed first, sits on top).
    option["xAxis"] = echarts.value_axis(money=True)
    option["yAxis"] = echarts.category_axis(projects, inverse=True)
    option["series"] = [
        {
            "type": "bar",
            "data": data_items,
            "emphasis": {"itemStyle": {"shadowBlur": 6, "shadowColor": "rgba(0,0,0,0.4)"}},
        }
    ]
    height = max(220, 30 * len(projects) + 80)
    return option, height


def _project_treemap_option(
    tokens: pd.DataFrame, sessions: pd.DataFrame, primary: str, color_map: dict[str, str]
) -> tuple[dict[str, Any], set[str]] | None:
    """Treemap of cost (project -> session); a leaf click drills that session.

    Each leaf's ``name`` is the **full** session_id (the click value), shown as a
    short 8-char label; ``nodeClick`` is disabled so a click drills instead of
    zooming. Returns the option plus the set of valid session ids (used to guard
    the drill against a project-header click).
    """
    col = data.cost_col(primary)
    per_session = _per_session_cost(tokens, primary)
    if per_session.empty:
        return None
    if not sessions.empty and "session_id" in sessions.columns:
        cols = [c for c in ["session_id", "project"] if c in sessions.columns]
        per_session = per_session.merge(sessions[cols], on="session_id", how="left")
    if "project" not in per_session.columns:
        per_session["project"] = "(unknown)"
    per_session["project"] = per_session["project"].fillna("(unknown)").replace("", "(unknown)")
    per_session = per_session[per_session[col] > 0].copy()
    if per_session.empty:
        return None

    c = echarts.colors()
    nodes: list[dict[str, Any]] = []
    session_ids: set[str] = set()
    for project, group in per_session.groupby("project"):
        proj = str(project)
        color = color_map.get(proj, "#9CA3AF")
        children = []
        for _, row in group.iterrows():
            sid = str(row["session_id"])
            session_ids.add(sid)
            children.append(
                {
                    "name": sid,
                    "value": round(float(row[col]), 2),
                    "label": {"show": True, "formatter": sid[:8]},
                    "itemStyle": {"color": color, "borderColor": c["grid"], "borderWidth": 1},
                }
            )
        nodes.append(
            {
                "name": proj,
                "value": round(float(group[col].sum()), 2),
                "itemStyle": {"color": color},
                "children": children,
            }
        )

    option = echarts.base_option()
    option.pop("grid", None)
    option["legend"] = {"show": False}
    option["title"] = _title(f"Cost treemap — project / session ({primary})")
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "formatter": echarts.js(
            "function(p){var t=p.treePathInfo||[];"
            "var proj=t.length>1?t[1].name:p.name;"
            "var leaf=t.length>2?t[2].name:'';"
            "var s=leaf?proj+' / '+leaf.slice(0,8):proj;"
            "return s+'<br/>$'+Number(p.value).toFixed(2);}"
        ),
    }
    option["series"] = [
        {
            "type": "treemap",
            "roam": False,
            "nodeClick": False,
            "breadcrumb": {"show": False},
            "top": 48,
            "data": nodes,
            "upperLabel": {"show": True, "height": 22, "color": c["text"]},
            "label": {"color": "#0f172b", "fontSize": 11},
            "levels": [
                {"itemStyle": {"gapWidth": 2, "borderColor": c["grid"], "borderWidth": 2}},
                {"itemStyle": {"gapWidth": 1, "borderColorSaturation": 0.4}},
            ],
        }
    ]
    return option, session_ids


def _apply_treemap_drill(value: Any, session_ids: set[str]) -> None:
    """Clicked session tile -> open that session in the Prompt Explorer (sticky-guarded).

    The on-page drill-down moved to the Prompt Explorer; a tile click now deep-links
    there with the session pre-selected. The "applied" marker keeps the sticky
    component value from bouncing back to the Prompt Explorer every time you return.
    """
    if not isinstance(value, str) or value not in session_ids:
        return
    if value == st.session_state.get(_XF_TREEMAP_APPLIED):
        return
    st.session_state[_XF_TREEMAP_APPLIED] = value
    st.session_state[DRILL_KEY] = value
    st.switch_page("pages/11_explorer.py")


def _prompts_per_session_option(counts: pd.DataFrame, n_sessions: int) -> dict[str, Any]:
    """Discrete bar: how many sessions have each prompt count.

    A click on a bar returns its category (the prompt count as a string) and is
    turned into the global :data:`filters.XF_PROMPT_COUNT` drill by the caller.
    """
    c = echarts.colors()
    vc = counts["prompt_count"].value_counts().sort_index()
    x = [str(int(i)) for i in vc.index.tolist()]
    y = [int(v) for v in vc.values.tolist()]
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = _title(f"Prompts per session (n = {n_sessions} sessions)")
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["xAxis"] = {
        **echarts.category_axis(x),
        "name": "Prompts in session",
        "nameLocation": "middle",
        "nameGap": 30,
        "nameTextStyle": {"color": c["muted"]},
    }
    # nameLocation "middle" keeps "Sessions" centered on the axis instead of
    # sitting at the top where it would collide with the chart title.
    option["yAxis"] = {
        **echarts.value_axis(name="Sessions"),
        "nameLocation": "middle",
        "nameGap": 36,
    }
    option["series"] = [
        {
            "type": "bar",
            "data": y,
            "itemStyle": {"color": theme.PALETTE[1], "borderRadius": [3, 3, 0, 0]},
            "label": {"show": True, "position": "top", "color": c["muted"]},
        }
    ]
    return option


def _cost_by_model_box_option(
    session_cost: pd.DataFrame, model_map: pd.DataFrame, cost: str, primary: str
) -> tuple[dict[str, Any], str] | None:
    """Box-and-whisker of per-session cost by dominant model (emits ``KEY_MODELS``).

    Long-tailed cost data blows the y-range out (a handful of huge sessions
    flatten every box), so the whiskers are the **p5–p95** range (not min/max)
    and the y-axis is **clipped** just above the largest p95; the few sessions
    beyond are acknowledged in the caption rather than stretching the scale.
    Boxes are ordered by median so the spread reads left-to-right; each carries
    its model's color. Returns the option plus a ready caption.
    """
    merged = session_cost.merge(model_map, on="session_id", how="left").dropna(subset=["model"])
    if merged.empty:
        return None
    rows: list[tuple[str, list[float], float]] = []
    groups: list[Any] = []
    for model, group in merged.groupby("model"):
        costs = group[cost].astype(float).to_numpy()
        if costs.size == 0:
            continue
        stats = data.box_stats(costs)  # p5/p95 whiskers
        rows.append((str(model), stats, stats[2]))
        groups.append(costs)
    if not rows:
        return None
    rows.sort(key=lambda r: r[2])  # by median
    models = [r[0] for r in rows]
    color_map = theme.model_color_map(models)

    # Clip the axis just above the tallest p95 whisker; count what spills over.
    y_max, n_above = data.box_cap(groups, [r[1] for r in rows])

    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = _title(f"Per-session cost by model ({primary})")
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
    y_axis = echarts.value_axis(money=True)
    if y_max is not None:
        y_axis["max"] = y_max
    option["yAxis"] = y_axis
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

    caption = (
        f"👆 Click a model to filter · n = {len(merged)} sessions · "
        "box = p25–median–p75, whiskers p5–p95"
    )
    if y_max is not None and n_above:
        caption += f" · {n_above} session(s) above ${y_max:,.2f} (axis clipped)"
    return option, caption


def main() -> None:
    """Render the Sessions page."""
    st.title("Sessions")

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all)
    tokens = frames.get("tokens", pd.DataFrame())
    prompts = frames.get("prompts", pd.DataFrame())
    sessions = frames.get("sessions", pd.DataFrame())
    primary = data.primary_provider()
    cost = data.cost_col(primary)

    if prompts.empty or "session_id" not in prompts.columns:
        st.info("No data for the current filters.")
        st.stop()

    session_cost = _per_session_cost(tokens, primary)
    model_map = data.dominant_model_per_session(prompts)

    # One project->color map built from the *unfiltered* universe (frames_all),
    # shared by the pareto, treemap and scatter: that is what keeps a project's
    # hue identical across the three (they each filter a different subset) and
    # stable when a filter narrows the set. (5.2)
    project_universe: set[str] = set()
    for frame in (frames_all.get("prompts"), frames_all.get("sessions")):
        if frame is not None and "project" in frame.columns:
            project_universe |= {str(p) for p in frame["project"].dropna().unique()}
    project_colors = theme.project_color_map(project_universe)

    theme.section(
        "Where the money goes",
        "Click a project bar to filter every page to it (Reset clears it); "
        "click a treemap session tile to open its drill-down below.",
    )
    pareto = _project_pareto_option(tokens, primary, project_colors)
    if pareto is not None:
        option, height = pareto
        clicked = echarts.render(option, key="sessions_pareto", height=f"{height}px", click=True)
        echarts.apply_click(clicked, filters.KEY_PROJECTS, synthetic=_SYNTHETIC)
    treemap = _project_treemap_option(tokens, sessions, primary, project_colors)
    if treemap is not None:
        option, session_ids = treemap
        clicked = echarts.render(option, key="sessions_treemap", height="440px", click=True)
        _apply_treemap_drill(clicked, session_ids)

    theme.section("Session economics")
    counts = prompts.groupby("session_id").size().reset_index(name="prompt_count")
    n_sessions = len(counts)

    left, right = st.columns(2)
    with left:
        clicked = echarts.render(
            _prompts_per_session_option(counts, n_sessions),
            key="sessions_per_session",
            height="360px",
            click=True,
        )
        filters.apply_prompt_count_click(clicked)
        st.caption(
            "👆 Click a bar to filter every page to the sessions with that many "
            "prompts (Reset clears it)."
        )
    with right:
        box = _cost_by_model_box_option(session_cost, model_map, cost, primary)
        if box is None:
            st.info("No per-session model data available.")
        else:
            option, box_caption = box
            clicked = echarts.render(option, key="sessions_cost_box", height="360px", click=True)
            echarts.apply_click(clicked, filters.KEY_MODELS)
            st.caption(box_caption)

    st.caption(
        "👆 Click a treemap session tile to open that session's day → session → "
        "prompt detail in the **Prompt Explorer**. Apply a filter on any chart, then "
        "use the **Explore →** button in the filter badge to inspect the selection."
    )


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
