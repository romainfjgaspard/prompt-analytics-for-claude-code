"""Shared global filter helpers for the dashboard app and its pages.

Filter selections live in ``st.session_state`` under stable keys so every page
reads the same global filter bar. The sidebar itself is rendered by
:func:`render_sidebar` on **every** page (5.3), not just the landing page, so a
filtered chart is never shown without the widgets that produced it; a
:func:`render_active_filter_badge` summarizes what is active with a one-click
reset. ``apply_filters`` filters the ``prompts`` frame by model, project,
category and date range, then cascades to ``tokens`` (by the surviving
``prompt_id`` values) and ``sessions`` (by surviving ``session_id``).

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

# Session-state keys for the global filter selections.
KEY_DATE_RANGE = "flt_date_range"
KEY_MODELS = "flt_models"
KEY_PROJECTS = "flt_projects"
KEY_CATEGORIES = "flt_categories"

_FILTER_KEYS = (KEY_DATE_RANGE, KEY_MODELS, KEY_PROJECTS, KEY_CATEGORIES)

# The sidebar multiselects own their widget keys (KEY_MODELS/...), so a chart
# click or the Reset button cannot write those keys directly -- Streamlit forbids
# mutating a widget-keyed value after the widget is instantiated in the same run.
# They stage the change under these keys instead; :func:`_drain_pending` applies
# it at the top of the next ``render_sidebar``, before the widgets are created.
_PENDING_SUFFIX = "__pending"
_PENDING_RESET = "_flt_pending_reset"


def stage_filter(filter_key: str, value: list[Any]) -> None:
    """Stage a cross-filter write to be applied before the sidebar renders.

    Called by :func:`echarts.apply_click` (a chart click) because the target is a
    multiselect widget key that cannot be mutated post-instantiation. The value is
    drained into the real key on the next run by :func:`_drain_pending`.
    """
    st.session_state[filter_key + _PENDING_SUFFIX] = value


def _drain_pending() -> None:
    """Apply staged cross-filter / reset writes (top of :func:`render_sidebar`)."""
    if st.session_state.pop(_PENDING_RESET, False):
        for k in _FILTER_KEYS:
            st.session_state.pop(k, None)
    for k in _FILTER_KEYS:
        pending = k + _PENDING_SUFFIX
        if pending in st.session_state:
            st.session_state[k] = st.session_state.pop(pending)


def _style_sidebar_tags() -> None:
    """Tone down the multiselect chips (the brief's "aggressive sidebar tags").

    Streamlit colors selected chips with the primary accent (the brand coral),
    which shouted next to the data. We repaint them a calm, translucent blue that
    reads on both themes (text inherits the theme color), and make the remove "x"
    a muted grey that turns danger-red on hover. Targets BaseWeb's stable
    ``data-baseweb="tag"`` node, scoped to the sidebar so charts are untouched.
    """
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] span[data-baseweb="tag"] {
            background-color: rgba(59, 130, 246, 0.16);
            border: 1px solid rgba(59, 130, 246, 0.45);
            color: inherit;
        }
        section[data-testid="stSidebar"] span[data-baseweb="tag"] span { color: inherit; }
        section[data-testid="stSidebar"] span[data-baseweb="tag"] svg { fill: #94A3B8; }
        section[data-testid="stSidebar"] span[data-baseweb="tag"] svg:hover { fill: #EF4444; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    """Read the current global filter selections from ``st.session_state``.

    Returns a dict with keys ``date_range`` (tuple[date, date] | None),
    ``models`` / ``projects`` / ``categories`` (list[str] | None). ``None``
    means "no restriction" (all values pass).
    """
    return {
        "date_range": st.session_state.get(KEY_DATE_RANGE),
        "models": st.session_state.get(KEY_MODELS),
        "projects": st.session_state.get(KEY_PROJECTS),
        "categories": st.session_state.get(KEY_CATEGORIES),
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

        if models and "model" in prompts.columns:
            mask &= prompts["model"].isin(models)

        if projects and "project" in prompts.columns:
            mask &= prompts["project"].isin(projects)

        if categories and "category" in prompts.columns:
            mask &= prompts["category"].isin(categories)

        if date_range and "timestamp" in prompts.columns:
            start = _to_date(date_range[0])
            end = _to_date(date_range[-1])
            if start is not None or end is not None:
                pdates = pd.to_datetime(prompts["timestamp"], errors="coerce").dt.date
                if start is not None:
                    mask &= pdates >= start
                if end is not None:
                    mask &= pdates <= end

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
    # Apply any staged cross-filter / reset writes before the widgets below are
    # instantiated (they own their keys and cannot be mutated afterwards).
    _drain_pending()
    st.sidebar.header("Filters")
    _style_sidebar_tags()
    opts = _options(frames)

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

    # Long value lists are collapsed by default (a sidebar full of "all selected"
    # chips was noise); the expander label summarises the selection at a glance.
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
    """A collapsed sidebar multiselect; the header shows ``all`` or ``n of N``.

    ``format_func`` (e.g. ``theme.model_label``) only changes the *displayed*
    text; the stored selection stays the raw option value, so the cross-filter
    keeps matching the underlying data column.
    """
    if not options:
        return
    # The multiselect owns its value via ``key=key`` (single source of truth):
    # Streamlit applies the new widget state *before* re-running the script, so
    # the summary in the expander label is always current. Mixing ``default=``
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
    with st.sidebar.expander(f"{title} — {summary}", expanded=False):
        st.multiselect(
            title,
            options=options,
            key=key,
            label_visibility="collapsed",
            format_func=format_func if format_func else str,
        )


def _active_parts(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Human summary fragments for each filter that restricts the data.

    A selection counts as active only when it is a proper, non-empty subset of
    what is available (an empty multiselect means "all" per ``apply_filters``).
    """
    state = get_filter_state()
    opts = _options(frames)
    parts: list[str] = []

    for key, label in (("models", "model"), ("projects", "project"), ("categories", "category")):
        selected = state[key]
        available = opts[key]
        if selected and available and 0 < len(selected) < len(available):
            noun = label if len(selected) == 1 else f"{label}s"
            parts.append(f"{len(selected)} {noun}")

    bounds = available_date_bounds(frames)
    date_range = state["date_range"]
    if bounds is not None and date_range:
        start = _to_date(date_range[0])
        end = _to_date(date_range[-1])
        if (start is not None and start > bounds[0]) or (end is not None and end < bounds[1]):
            lo = start.isoformat() if start else "…"
            hi = end.isoformat() if end else "…"
            parts.append(f"{lo} → {hi}")
    return parts


def reset_filters() -> None:
    """Stage clearing all global filter selections.

    Staged (not popped here) because the Reset button fires after the sidebar
    multiselects are already instantiated this run; the actual pop happens in
    :func:`_drain_pending` at the top of the next ``render_sidebar``.
    """
    st.session_state[_PENDING_RESET] = True


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
    """
    parts = _active_parts(frames)
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
    if button.button("Reset", help="Clear all filters", width="stretch"):
        reset_filters()
        st.rerun()
