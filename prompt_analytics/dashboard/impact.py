"""Global before/after date-pivot mode for the dashboard (Axe E / DASH2).

A sidebar toggle turns the whole dashboard into *comparison mode*: pick a "switch
date" (when you installed something, scoped a CLAUDE.md, switched a model) and the
wired views (Usage, Composition) reframe themselves as **before vs after + delta**.

The comparison is deliberately shown in **workload-normalized ratios** (cost per
prompt, output cost share, context rent share, cache read per turn, output tokens
per prompt), not raw totals: a raw before/after is confounded by how much you
worked: bill more after and the optimization looks like it *cost* you. The
ratios isolate the config from the workload; the workload confounders (volume,
depth, task mix) ride alongside, in an expander, so the deltas are never over-sold
as causal. This is an observational split, not a controlled experiment.

Every number comes from :func:`analytics.impact_report` -- the same shared
``ImpactReport`` the CLI ``impact`` command prints -- so the table and these
cards can never drift. The state lives in ``st.session_state`` (owned by the
sidebar widgets, persisted across page navigation by :func:`filters.render_sidebar`).
The pivot keys mirror the filter keys' persistence so a selection survives a hop
through a page that hides the sidebar.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from prompt_analytics import analytics
from prompt_analytics.analytics import _day_before, _impact_fmt_change, _impact_fmt_value
from prompt_analytics.dashboard import data

# Sidebar widget keys (owned via ``key=``); persisted like the filter keys so the
# compare-mode selection survives page navigation (st.navigation garbage-collects
# keyed widget state on the run that leaves a page otherwise).
KEY_PIVOT_ON = "flt_pivot_on"
KEY_PIVOT_DATE = "flt_pivot_date"
PIVOT_KEYS = (KEY_PIVOT_ON, KEY_PIVOT_DATE)

# Sentinel for the "no suggestion picked" row of the suggestions selectbox.
_PICK_NONE = "— pick a date —"
_PICK_KEY = "flt_pivot_suggest"
# Sticky guard: the last suggestion we applied to the date input, so re-selecting
# the same one (every rerun) never clobbers a date the user typed afterwards.
_PICK_APPLIED = "_flt_pivot_suggest_applied"


def current_pivot() -> str | None:
    """The active pivot day (``YYYY-MM-DD``) when compare mode is on, else ``None``.

    Read by every wired page to decide whether to reframe itself. Returns ``None``
    whenever the toggle is off or no valid date is stored, so a page can guard on
    a single truthy check.
    """
    if not st.session_state.get(KEY_PIVOT_ON):
        return None
    value = st.session_state.get(KEY_PIVOT_DATE)
    day = _to_iso(value)
    return day


def _to_iso(value: object) -> str | None:
    """Best-effort ``YYYY-MM-DD`` from a date / datetime / string (else ``None``)."""
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        ts = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(ts) else ts.date().isoformat()
    return None


@st.cache_data(show_spinner=False)
def _suggested_pivots(data_dir_str: str, cache_key: str) -> list[tuple[str, str]]:
    """Cache-keyed wrapper over :func:`analytics.suggest_pivots` (mtime-keyed)."""
    ds = analytics.dataset_from_csvs(Path(data_dir_str))
    return analytics.suggest_pivots(ds)


def render_pivot_controls(frames: dict[str, pd.DataFrame]) -> None:
    """Sidebar "Compare before/after a change" toggle + date + auto-suggestions.

    Rendered by :func:`filters.render_sidebar` on every page. The date input is
    bounded to the data range; a selectbox offers the config-change dates detected
    by :func:`analytics.suggest_pivots` (mtime of CLAUDE.md / settings.json) as a
    typing aid -- picking one fills the date, but the analysis never depends on it.
    """
    from prompt_analytics.dashboard import filters

    bounds = filters.available_date_bounds(frames)
    st.sidebar.divider()
    st.sidebar.checkbox(
        "📊 Compare before/after a change",
        key=KEY_PIVOT_ON,
        help="Split the history on a switch date and show before vs after in "
        "workload-normalized ratios (cost per prompt, output share, context rent…).",
    )
    if not st.session_state.get(KEY_PIVOT_ON):
        return
    if bounds is None:
        st.sidebar.info("No dates available to split on.")
        return
    min_d, max_d = bounds

    # Auto-suggestions: a picked one drives the date input; "pick a date" leaves
    # the user's manual choice untouched.
    directory = data.data_dir()
    suggestions = _suggested_pivots(str(directory), data._mtimes_key(directory))
    if suggestions:
        labels = {f"{day} · {label}": day for day, label in suggestions}
        choice = st.sidebar.selectbox(
            "Detected config changes",
            [_PICK_NONE, *labels.keys()],
            key=_PICK_KEY,
            help="Dates a CLAUDE.md / settings.json was last modified — a likely "
            "'I changed my setup here'. Only a typing aid; the split is whatever "
            "date you set below.",
        )
        picked = labels.get(choice)
        # Apply a suggestion only when it *changes* (a sticky guard), so a date the
        # user typed after picking one isn't clobbered on the next rerun.
        if picked and st.session_state.get(_PICK_APPLIED) != picked:
            day = _to_iso(picked)
            if day is not None:
                st.session_state[KEY_PIVOT_DATE] = _dt.date.fromisoformat(day)
            st.session_state[_PICK_APPLIED] = picked
        elif not picked:
            st.session_state.pop(_PICK_APPLIED, None)

    # Seed / clamp the stored pivot to the data range (default to the midpoint so
    # both sides are non-empty out of the box).
    cur = _to_iso(st.session_state.get(KEY_PIVOT_DATE))
    cur_d = _dt.date.fromisoformat(cur) if cur else None
    if cur_d is None or not (min_d <= cur_d <= max_d):
        st.session_state[KEY_PIVOT_DATE] = min_d + (max_d - min_d) / 2
    st.sidebar.date_input(
        "Switch date (pivot)",
        min_value=min_d,
        max_value=max_d,
        key=KEY_PIVOT_DATE,
        help="The pivot day is counted in the AFTER side (before = up to the day before it).",
    )


def _delta_metric(col: Any, metric: analytics.ImpactMetric) -> None:
    """One before/after card: the AFTER value, with the change as a grey delta.

    The big number is the AFTER value; the delta is the change from BEFORE (the
    same string the CLI prints). Colour is always ``off`` (grey): up isn't
    universally "good" here (cost per prompt up is bad, output share up is
    neutral), so we never imply a verdict -- the honesty ADN of the whole product.
    The BEFORE value rides in the help tooltip.
    """
    after = _impact_fmt_value(metric.after, metric.fmt)
    before = _impact_fmt_value(metric.before, metric.fmt)
    change = _impact_fmt_change(metric.before, metric.after, metric.fmt)
    col.metric(
        metric.label,
        after,
        delta=None if change == "-" else change,
        delta_color="off",
        help=f"Before: {before}",
    )


def render_impact_panel(ds: analytics.Dataset, provider: str, pivot: str) -> None:
    """The canonical before/after panel: normalized ratio cards + confounders + note.

    Built entirely from :func:`analytics.impact_report` so it stays identical to
    the CLI ``impact`` table. Used at the top of every wired page in compare mode.
    """
    report = analytics.impact_report(ds, provider=provider, pivot=pivot)
    pivot_before = _day_before(pivot)

    st.caption(
        f"**Before {pivot}:** {report.before_prompts:,} prompts over "
        f"{report.before_days} active days (up to {pivot_before})  ·  "
        f"**After:** {report.after_prompts:,} prompts over {report.after_days} active days."
    )
    if not report.has_both_sides:
        empty = "before" if report.before_prompts == 0 else "after"
        st.warning(
            f"No prompts **{empty}** the pivot — pick a date inside the data range for a "
            "meaningful comparison."
        )

    ratios = [m for m in report.metrics if not m.confounder]
    cols = st.columns(len(ratios))
    for col, metric in zip(cols, ratios, strict=False):
        _delta_metric(col, metric)

    confounders = [m for m in report.metrics if m.confounder]
    if confounders:
        with st.expander("Workload confounders — how the workload itself moved (not the change)"):
            ccols = st.columns(len(confounders))
            for col, metric in zip(ccols, confounders, strict=False):
                _delta_metric(col, metric)

    st.caption(
        "Ratios are **workload-normalized** (per prompt, per turn, or as a cost share) so the "
        "change reads through the workload; the confounders describe how the workload itself "
        "moved. This is an **observational split, not a controlled experiment** — if volume, "
        "depth or task mix shifted a lot, read the deltas as correlation, not proven causation. "
        "👉 Same split on the command line: `prompt-analytics impact --pivot " + pivot + "`."
    )
