"""Shared global filter helpers for the dashboard app and its pages.

Filter selections live in ``st.session_state`` under stable keys so every page
reads the same global filter bar. The sidebar itself is rendered by
:func:`render_sidebar` on **every** page (5.3), not just the landing page, so a
filtered chart is never shown without the widgets that produced it.

There are two layers, deliberately kept separate:

* the **sidebar** selections (``KEY_*``) are *persistent* — owned by the sidebar
  widgets, changed only there, never reset or surfaced in the badge;
* the **chart-click drill** (``XF_*``) is a transient cross-filter applied by
  clicking a bar/treemap/brush. It alone raises the :func:`render_active_filter_badge`
  "Filtered: …" badge and is the only thing its Reset button clears.

``apply_filters`` ANDs both layers: it filters the ``prompts`` frame by the
sidebar model/project/category/date selection *and* the drill, then cascades to
``tokens`` (by the surviving ``prompt_id`` values) and ``sessions`` (by surviving
``session_id``).

Pseudo-prompts (session overhead such as ``:_continuation``) exist in
``tokens.csv`` only, never in ``prompts.csv``: their token rows are cascaded
through their **session** instead, so the dashboard totals reconcile with the
CLI's ``summary`` (which includes them).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pandas as pd
import streamlit as st

from prompt_analytics.dashboard import theme

# Session-state keys for the persistent sidebar filter selections. The sidebar
# widgets own these (via ``key=``); they survive page navigation and are *only*
# changed by the user in the sidebar -- a chart click or Reset never touches them.
KEY_DATE_RANGE = "flt_date_range"
KEY_MODELS = "flt_models"
KEY_PROJECTS = "flt_projects"
KEY_CATEGORIES = "flt_categories"

_FILTER_KEYS = (KEY_DATE_RANGE, KEY_MODELS, KEY_PROJECTS, KEY_CATEGORIES)

# Cross-filter (chart-click "drill") keys, distinct from the sidebar keys above.
# A bar/treemap/brush click writes one of these to narrow the dashboard *on top
# of* the sidebar selection. They are plain session-state keys (no widget owns
# them), so they can be written and cleared directly -- and they alone drive the
# "Filtered: …" badge and its Reset button. This split is what makes the sidebar
# filters persistent: Reset clears the drill, never the sidebar.
XF_DATE_RANGE = "xf_date_range"
XF_MODELS = "xf_models"
XF_PROJECTS = "xf_projects"
XF_CATEGORIES = "xf_categories"

_XF_KEYS = (XF_DATE_RANGE, XF_MODELS, XF_PROJECTS, XF_CATEGORIES)

# The Explorer's local day / session focus, also cleared by Reset (a treemap /
# top-10 tile click sets ``drill_session``; ``_xf_treemap_applied`` is the
# treemap's sticky-value guard, cleared so a re-click after Reset fires again).
_DRILL_KEYS = ("drill_date", "drill_session", "_xf_treemap_applied")

# Map a sidebar widget key (what the chart-click callers still reference) to its
# cross-filter twin. A click writes the twin, never the sidebar key.
_XF_FOR = {
    KEY_DATE_RANGE: XF_DATE_RANGE,
    KEY_MODELS: XF_MODELS,
    KEY_PROJECTS: XF_PROJECTS,
    KEY_CATEGORIES: XF_CATEGORIES,
}


def xf_key_for(sidebar_key: str) -> str:
    """The cross-filter (chart-click) twin of a sidebar widget key.

    Called by :func:`echarts.apply_click` / :func:`echarts.apply_date_range` so a
    chart click lands on the drill key instead of the persistent sidebar key.
    """
    return _XF_FOR.get(sidebar_key, sidebar_key)


def _css_escape(text: str) -> str:
    """Escape a string for use inside a CSS attribute selector's double quotes."""
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _filter_tag_colors(
    frames: dict[str, pd.DataFrame], opts: dict[str, list[Any]]
) -> dict[str, str]:
    """Map each multiselect chip's *displayed* label to its semantic color.

    Mirrors the chart palettes (``theme.model_color_map`` / ``project_color_map``
    / ``CATEGORY_COLORS``) so a chip reads with the same hue the value carries in
    the Models / Sessions / Prompts charts. Keyed by the label as it appears in
    the chip — i.e. the ``format_func`` output for models (``model_label``), the
    raw value for projects / categories — because that is the text the CSS
    selector matches.
    """
    colors: dict[str, str] = {}

    model_colors = theme.model_color_map(opts["models"])
    for model, color in model_colors.items():
        colors[theme.model_label(model)] = color

    # Build the project hue map from the *same* unfiltered universe the Sessions
    # tab uses, so a project's chip and its treemap tile share one color.
    universe: set[str] = set()
    for frame in (frames.get("prompts"), frames.get("sessions")):
        if frame is not None and "project" in frame.columns:
            universe |= {str(p) for p in frame["project"].dropna().unique()}
    project_colors = theme.project_color_map(universe)
    for project in opts["projects"]:
        if project in project_colors:
            colors[str(project)] = project_colors[project]

    for category in opts["categories"]:
        if category in theme.CATEGORY_COLORS:
            colors[str(category)] = theme.CATEGORY_COLORS[category]

    return colors


def _style_filter_tags(frames: dict[str, pd.DataFrame], opts: dict[str, list[Any]]) -> None:
    """Color each selected filter chip with its value's semantic hue.

    Streamlit paints selected chips with the primary accent (the brand coral),
    which (a) shouted next to the data and (b) carried no meaning. Instead we give
    each chip the color the value owns in the charts -- a darker clay for Fable, a
    lighter one for Sonnet, each project's distinct hue, etc. -- as a translucent
    fill + solid border, text inheriting the theme color, and the remove "x" a
    muted grey that turns danger-red on hover.

    The rules are **unscoped** (any ``span[data-baseweb="tag"]``) on purpose: the
    chips now live in a :func:`st.popover`, whose body renders outside the sidebar
    DOM, so the old ``section[data-testid="stSidebar"]`` scope no longer reached
    them. The multiselects are the only tag-emitting widgets in the app, so a
    global rule is safe. Each per-value rule targets the chip by its label text
    via ``:has([title="…"])`` (the chip's inner node carries the label as its
    ``title``); a calm slate fallback covers any label without a mapped color.
    """
    rules = [
        # Calm slate fallback for any chip (and the base box-model / "x" styling).
        'span[data-baseweb="tag"]{'
        "background-color:rgba(100,116,139,0.18);"
        "border:1px solid rgba(100,116,139,0.5);color:inherit;}",
        'span[data-baseweb="tag"] span{color:inherit;}',
        'span[data-baseweb="tag"] svg{fill:#94A3B8;}',
        'span[data-baseweb="tag"] svg:hover{fill:#EF4444;}',
    ]
    for label, hex_color in _filter_tag_colors(frames, opts).items():
        sel = f'span[data-baseweb="tag"]:has([title="{_css_escape(label)}"])'
        rules.append(
            f"{sel}{{background-color:{theme._rgba(hex_color, 0.22)};"
            f"border:1px solid {theme._rgba(hex_color, 0.85)};}}"
        )
    st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)


def persist_filters() -> None:
    """Keep the global filter selections alive on pages that hide the sidebar.

    The selections live under widget keys, which Streamlit garbage-collects on a
    run where no widget claims them. Pages without the filter sidebar (Optimize,
    Quotas, How it works) call this so a selection made elsewhere survives a
    detour through them -- re-assigning each key marks it as user-owned, the same
    cleanup-exemption :func:`render_sidebar` relies on for tab navigation.
    """
    for k in _FILTER_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def get_filter_state() -> dict[str, Any]:
    """Read the current filter selections from ``st.session_state``.

    Returns the persistent sidebar selections under ``date_range`` /
    ``models`` / ``projects`` / ``categories`` and the chart-click drill under
    the ``xf_*`` keys (``xf_date_range`` / ``xf_models`` / …). ``None`` means "no
    restriction" (all values pass). :func:`apply_filters` ANDs the two together.
    """
    return {
        "date_range": st.session_state.get(KEY_DATE_RANGE),
        "models": st.session_state.get(KEY_MODELS),
        "projects": st.session_state.get(KEY_PROJECTS),
        "categories": st.session_state.get(KEY_CATEGORIES),
        "xf_date_range": st.session_state.get(XF_DATE_RANGE),
        "xf_models": st.session_state.get(XF_MODELS),
        "xf_projects": st.session_state.get(XF_PROJECTS),
        "xf_categories": st.session_state.get(XF_CATEGORIES),
    }


def _to_date(value: object) -> _dt.date | None:
    """Best-effort conversion of a value to a ``datetime.date``."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    ts = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def _restricts(selected: list[Any] | None, available: list[Any]) -> bool:
    """Whether a multiselect selection actually narrows the data.

    A selection that covers *every* available value (or is empty) means "all":
    applying it as a filter would needlessly drop rows whose value is blank/NaN
    and therefore absent from ``available`` -- e.g. a model-less prompt with no
    billed assistant turn, which ``_options`` strips from the chips. Counting it
    out of the totals on a default (everything-selected) view is exactly the
    KPI/extract mismatch we want to avoid, so only a *proper* subset filters.
    """
    if not selected:
        return False
    return not set(available) <= set(selected)


def _date_mask(prompts: pd.DataFrame, date_range: tuple[Any, Any] | None) -> pd.Series:
    """A boolean mask of ``prompts`` whose timestamp falls within ``date_range``.

    All-``True`` when there is no usable range / column, so it composes with
    ``&=`` for both the sidebar date range and the chart-brush drill.
    """
    mask = pd.Series(True, index=prompts.index)
    if not date_range or "timestamp" not in prompts.columns:
        return mask
    start = _to_date(date_range[0])
    end = _to_date(date_range[-1])
    if start is None and end is None:
        return mask
    pdates = pd.to_datetime(prompts["timestamp"], errors="coerce").dt.date
    if start is not None:
        mask &= pdates >= start
    if end is not None:
        mask &= pdates <= end
    return mask


def apply_filters(
    frames: dict[str, pd.DataFrame],
    *,
    date_range: tuple[Any, Any] | None = None,
    models: list[str] | None = None,
    projects: list[str] | None = None,
    categories: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Filter ``prompts`` then cascade to ``tokens`` and ``sessions``.

    Args:
        frames: The dict returned by :func:`dashboard.data.load_all`.
        date_range: Optional ``(start, end)`` inclusive date bounds. If omitted,
            the selection is read from ``st.session_state``.
        models: Optional list of allowed ``model`` values (``None`` = all).
        projects: Optional list of allowed ``project`` values (``None`` = all).
        categories: Optional list of allowed ``category`` values (``None`` = all).

    Returns:
        A new dict of filtered copies with the same keys as ``frames``.
    """
    # Fall back to session-state selections when not explicitly provided.
    state = get_filter_state()
    if date_range is None:
        date_range = state["date_range"]
    if models is None:
        models = state["models"]
    if projects is None:
        projects = state["projects"]
    if categories is None:
        categories = state["categories"]

    prompts = frames.get("prompts", pd.DataFrame()).copy()
    tokens = frames.get("tokens", pd.DataFrame()).copy()
    sessions = frames.get("sessions", pd.DataFrame()).copy()

    if not prompts.empty:
        mask = pd.Series(True, index=prompts.index)
        opts = _options(frames)

        if _restricts(models, opts["models"]) and "model" in prompts.columns:
            mask &= prompts["model"].isin(models)

        if _restricts(projects, opts["projects"]) and "project" in prompts.columns:
            mask &= prompts["project"].isin(projects)

        if _restricts(categories, opts["categories"]) and "category" in prompts.columns:
            mask &= prompts["category"].isin(categories)

        mask &= _date_mask(prompts, date_range)

        # AND the chart-click drill on top of the sidebar selection. Unlike the
        # sidebar multiselects (where "everything selected" means "no filter",
        # handled by ``_restricts``), a drill is always an explicit single-value
        # pick, so any non-empty selection narrows.
        xf_models = state.get("xf_models")
        if xf_models and "model" in prompts.columns:
            mask &= prompts["model"].isin(xf_models)
        xf_projects = state.get("xf_projects")
        if xf_projects and "project" in prompts.columns:
            mask &= prompts["project"].isin(xf_projects)
        xf_categories = state.get("xf_categories")
        if xf_categories and "category" in prompts.columns:
            mask &= prompts["category"].isin(xf_categories)
        mask &= _date_mask(prompts, state.get("xf_date_range"))

        prompts = prompts[mask]

    # Cascade: keep only tokens / sessions referenced by surviving prompts.
    surviving_prompt_ids = set(prompts["prompt_id"]) if "prompt_id" in prompts.columns else set()
    surviving_session_ids = set(prompts["session_id"]) if "session_id" in prompts.columns else set()

    if not tokens.empty and "prompt_id" in tokens.columns:
        keep = tokens["prompt_id"].isin(surviving_prompt_ids)
        # Pseudo-prompt rows (session overhead) have no prompts.csv row at all:
        # cascading them on prompt_id would drop them even with NO filter
        # active. They follow their session instead.
        original_prompts = frames.get("prompts", pd.DataFrame())
        real_prompt_ids = (
            set(original_prompts["prompt_id"]) if "prompt_id" in original_prompts.columns else set()
        )
        if "session_id" in tokens.columns:
            is_pseudo = ~tokens["prompt_id"].isin(real_prompt_ids)
            keep |= is_pseudo & tokens["session_id"].isin(surviving_session_ids)
        tokens = tokens[keep]

    if not sessions.empty and "session_id" in sessions.columns:
        sessions = sessions[sessions["session_id"].isin(surviving_session_ids)]

    result = dict(frames)
    result["prompts"] = prompts
    result["tokens"] = tokens
    result["sessions"] = sessions
    return result


# ---------------------------------------------------------------------------
# Shared sidebar (rendered on every page, 5.3) + active-filter badge.
# ---------------------------------------------------------------------------


def _unique_sorted(series: pd.Series) -> list[Any]:
    """Return sorted unique non-null values of a series."""
    if series is None or series.empty:
        return []
    values = series.dropna().unique().tolist()
    return sorted(values)


def available_date_bounds(
    frames: dict[str, pd.DataFrame],
) -> tuple[_dt.date, _dt.date] | None:
    """Derive (min_date, max_date) from prompts.timestamp or sessions.start_date.

    Both source columns are normalized to aware UTC by ``data.load_all`` (8.1),
    so concatenating them no longer raises the tz-naive/tz-aware comparison.
    """
    candidates: list[pd.Series] = []
    prompts = frames.get("prompts")
    if prompts is not None and "timestamp" in prompts.columns:
        candidates.append(pd.to_datetime(prompts["timestamp"], errors="coerce", utc=True))
    sessions = frames.get("sessions")
    if sessions is not None and "start_date" in sessions.columns:
        candidates.append(pd.to_datetime(sessions["start_date"], errors="coerce", utc=True))
    if not candidates:
        return None
    alldates = pd.concat(candidates).dropna()
    if alldates.empty:
        return None
    return alldates.min().date(), alldates.max().date()


def _options(frames: dict[str, pd.DataFrame]) -> dict[str, list[Any]]:
    """Available filter options derived from the *unfiltered* frames."""
    prompts = frames.get("prompts", pd.DataFrame())
    models = []
    if "model" in prompts.columns:
        # Drop blank/NaN model ids: they would render as an empty, unselectable
        # chip in the multiselect (some prompts carry no model).
        non_blank = prompts.loc[prompts["model"].astype(str).str.strip() != "", "model"]
        models = _unique_sorted(non_blank)
    projects = _unique_sorted(prompts["project"]) if "project" in prompts.columns else []
    categories: list[Any] = []
    if "category" in prompts.columns:
        cats = prompts["category"].astype("object")
        valid = cats.notna() & (cats.astype(str).str.strip() != "")
        valid &= cats.astype(str).str.lower() != "nan"
        categories = _unique_sorted(prompts.loc[valid, "category"])
    return {"models": models, "projects": projects, "categories": categories}


_REFRESH_MSG = "_refresh_msg"


def render_refresh_button() -> None:
    """A sidebar "Refresh data" button that re-runs the extract pipeline in place.

    Saves the terminal round-trip: on click it re-extracts prompts from the local
    ``~/.claude`` logs, snapshots the quota and re-categorizes (heuristic), clears
    the mtime-keyed data cache, then reruns so the fresh CSVs load immediately.
    Hidden on the demo dataset (no local logs, and it must never overwrite the
    committed ``demo_data``). The success summary is stashed in session_state and
    surfaced as a toast on the rerun (a toast shown right before ``st.rerun`` would
    be discarded).
    """
    from prompt_analytics.dashboard import data

    msg = st.session_state.pop(_REFRESH_MSG, None)
    if msg:
        st.toast(f"✅ {msg}")
    if data.is_demo():
        return
    if st.sidebar.button(
        "🔄 Refresh data",
        width="stretch",
        help="Re-extract prompts from ~/.claude and re-categorize, then reload",
    ):
        with st.spinner("Extracting & categorizing…"):
            try:
                summary = data.refresh_data()
            except Exception as exc:  # surface the failure, don't crash the page
                st.sidebar.error(f"Refresh failed: {exc}")
                return
        # mtimes change after extract, but clear the cache so the reload is
        # unconditional (st.cache_data.clear() drops every @st.cache_data entry,
        # i.e. both _load_cached and _load_prompt_texts_cached).
        st.cache_data.clear()
        st.session_state[_REFRESH_MSG] = summary
        st.rerun()


def render_sidebar(frames: dict[str, pd.DataFrame]) -> None:
    """Render the global filter bar and persist selections to session_state.

    Called by **every** page from the *unfiltered* frames so the option lists
    never shrink to the current selection. Selections are stored under the
    shared ``KEY_*`` keys, so navigating between pages keeps the same filter.
    """
    # Keep the multiselect selections across page navigation. A keyed widget's
    # state is otherwise garbage-collected by Streamlit when its page is left
    # (st.navigation), so the filters would reset on every tab change. Re-assigning
    # each key to itself marks it as user-owned and exempts it from that cleanup.
    for wkey in (KEY_MODELS, KEY_PROJECTS, KEY_CATEGORIES):
        if wkey in st.session_state:
            st.session_state[wkey] = st.session_state[wkey]
    render_refresh_button()
    st.sidebar.header("Filters")
    opts = _options(frames)
    _style_filter_tags(frames, opts)

    # --- Date range ---
    bounds = available_date_bounds(frames)
    if bounds is None:
        st.sidebar.info("No dates available.")
    else:
        min_d, max_d = bounds
        # The date_input owns ``KEY_DATE_RANGE`` via ``key=`` so it keeps the
        # *partial* selection between the two clicks of a range pick (the old
        # ``value=`` + manual assignment collapsed the first click into a degenerate
        # ``(x, x)`` range, so a second date could never be chosen). We only
        # normalize the stored value here (strings from a chart brush -> dates,
        # out-of-bounds -> the full range), which also re-commits it as the
        # cross-page persistence bounce. A 1- or 2-tuple of in-bounds dates (incl.
        # a mid-selection single pick) is preserved untouched.
        cur = st.session_state.get(KEY_DATE_RANGE)
        parts = list(cur) if isinstance(cur, (tuple, list)) else ([cur] if cur else [])
        dates = [_to_date(v) for v in parts][:2]
        if dates and all(d is not None and min_d <= d <= max_d for d in dates):
            st.session_state[KEY_DATE_RANGE] = tuple(dates)
        else:
            st.session_state[KEY_DATE_RANGE] = (min_d, max_d)
        st.sidebar.date_input(
            "Date range",
            min_value=min_d,
            max_value=max_d,
            key=KEY_DATE_RANGE,
        )

    # Long value lists stay tucked behind a popover (a sidebar full of "all
    # selected" chips was noise); the button label summarises the selection at a
    # glance and the popover stays open across check/uncheck reruns.
    _multiselect_filter("Model", KEY_MODELS, opts["models"], format_func=theme.model_label)
    _multiselect_filter("Project", KEY_PROJECTS, opts["projects"])
    _multiselect_filter("Category", KEY_CATEGORIES, opts["categories"])  # (5.4)

    # Demo banner at the very bottom of the sidebar (generic chrome that would
    # otherwise steal vertical space from the first chart in the main column).
    from prompt_analytics.dashboard import data

    data.render_demo_banner()


def _multiselect_filter(
    title: str,
    key: str,
    options: list[Any],
    *,
    format_func: Any | None = None,
) -> None:
    """A collapsed sidebar multiselect; the button shows ``all`` or ``n of N``.

    Lives in a :func:`st.popover` rather than an :func:`st.expander` on purpose:
    an expander re-applies its ``expanded=False`` default on every rerun, so each
    check/uncheck (which reruns the script) snapped it shut -- you could only
    toggle one value at a time. A popover stays open while you interact with the
    widgets inside it and closes only on an outside click, so several values can
    be toggled in one go.

    ``format_func`` (e.g. ``theme.model_label``) only changes the *displayed*
    text; the stored selection stays the raw option value, so the cross-filter
    keeps matching the underlying data column.
    """
    if not options:
        return
    # The multiselect owns its value via ``key=key`` (single source of truth):
    # Streamlit applies the new widget state *before* re-running the script, so
    # the summary in the popover label is always current. Mixing ``default=``
    # with a manual ``session_state`` assignment on a key-less widget lagged the
    # label by one rerun (it showed the previous run's selection). We seed the
    # state once (absent -> all selected) and prune any values that are no longer
    # available (the option list can shrink), both *before* the widget renders.
    if key not in st.session_state:
        st.session_state[key] = list(options)
    else:
        valid = [v for v in st.session_state[key] if v in options]
        if valid != list(st.session_state[key]):
            st.session_state[key] = valid
    current = st.session_state[key]
    summary = "all" if len(current) == len(options) else f"{len(current)} of {len(options)}"
    with st.sidebar.popover(f"{title} — {summary}", width="stretch"):
        st.multiselect(
            title,
            options=options,
            key=key,
            label_visibility="collapsed",
            format_func=format_func if format_func else str,
        )


def _xf_parts() -> list[str]:
    """Human summary fragments for the chart-click drill (the badge's content).

    Only the cross-filter (``xf_*``) selection is summarized: the persistent
    sidebar filters are deliberately *not* surfaced here, so the badge — and its
    Reset — concern only the drill the user applied by clicking a chart.
    """
    state = get_filter_state()
    parts: list[str] = []
    for m in state.get("xf_models") or []:
        parts.append(theme.model_label(m))
    for p in state.get("xf_projects") or []:
        parts.append(str(p))
    for c in state.get("xf_categories") or []:
        parts.append(str(c))

    xf_date = state.get("xf_date_range")
    if xf_date:
        start = _to_date(xf_date[0])
        end = _to_date(xf_date[-1])
        lo = start.isoformat() if start else "…"
        hi = end.isoformat() if end else "…"
        parts.append(f"{lo} → {hi}")
    return parts


def set_cross_filter(filter_key: str, value: list[Any]) -> bool:
    """Apply a chart-click drill to the cross-filter twin of ``filter_key``.

    Writes the drill key directly (no widget owns it) and returns ``True`` when
    the value actually changed — the caller (:mod:`echarts`) reruns only then, so
    the component's sticky value cannot re-apply the same pick every rerun.
    """
    xf_key = xf_key_for(filter_key)
    if st.session_state.get(xf_key) == value:
        return False
    st.session_state[xf_key] = value
    return True


def reset_filters() -> None:
    """Clear the chart-click drill (and the Explorer's day/session focus).

    The persistent sidebar selections are intentionally left untouched: Reset only
    undoes what a chart click applied. The drill keys are plain (non-widget)
    session-state, so they are popped directly here and take effect on the rerun
    the Reset button triggers.
    """
    for k in (*_XF_KEYS, *_DRILL_KEYS):
        st.session_state.pop(k, None)


def granularity_control(dates: pd.Series, *, key: str) -> str:
    """Day / Week / Month toggle, defaulting to a grain that fits the span.

    Shared by the time-series charts (app, Models) so they bucket identically;
    the default follows :func:`data.auto_granularity` (a single daily bar is
    useless over months) but the user can override it.
    """
    from prompt_analytics.dashboard import data

    default = data.auto_granularity(dates)
    choice = st.segmented_control(
        "Group by", options=list(data.GRANULARITIES), default=default, key=key
    )
    return str(choice) if choice else default


def render_active_filter_badge(
    frames: dict[str, pd.DataFrame], *, explore_link: bool = True
) -> None:
    """Show a "Filtered: …" badge with Explore + Reset buttons when a filter is active.

    Rendered on every page so a sub-sampled chart is never silently shown as if
    it were the whole dataset (the audit's "phantom filters" risk). Because it
    appears exactly when the user has narrowed the dashboard (by clicking charts),
    it is also where the **drill-through** lives: an *Explore →* button jumps to
    the Explorer page, which inspects the same selection as day → session → prompt
    detail. ``explore_link=False`` on the Explorer page itself (no self-link).

    Only the chart-click drill raises the badge; the persistent sidebar filters
    never do (they are changed only in the sidebar). ``frames`` is kept for the
    call-site signature but no longer read.
    """
    parts = _xf_parts()
    if not parts:
        return
    if explore_link:
        summary, explore, button = st.columns([4, 1, 1])
        if explore.button(
            "Explore →",
            help="Open the matching day / session / prompt detail in Explorer",
            width="stretch",
        ):
            st.switch_page("pages/11_explorer.py")
    else:
        summary, button = st.columns([5, 1])
    summary.info("🔎 Filtered: " + " · ".join(parts))
    if button.button("Reset", help="Clear the chart-click filter", width="stretch"):
        reset_filters()
        st.rerun()
