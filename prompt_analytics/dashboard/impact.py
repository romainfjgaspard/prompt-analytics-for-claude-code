"""Before/after date-pivot comparison for the dashboard (Axe E / DASH2).

A dedicated **Compare** page (``pages/9_compare.py``, after Optimize) turns the
history into a *before vs after* of a single switch date: pick the day you
installed something, scoped a CLAUDE.md, or changed a model, and the page reframes
the spine as **before (left) vs after (right)**.

The comparison is deliberately shown in **workload-normalized ratios** (cost per
prompt, output cost share, context rent share, cache read per turn, output tokens
per prompt) and **averages**, never raw totals: a raw before/after is confounded
by how much you worked (bill more after and the optimization looks like it *cost*
you). The ratios isolate the config from the workload; the workload confounders
(volume, depth, task mix) ride alongside, in an expander, so the deltas are never
over-sold as causal. This is an observational split, not a controlled experiment.

Every ratio comes from :func:`analytics.impact_report` -- the same shared
``ImpactReport`` the CLI ``impact`` command prints -- so the cards and the CLI
table can never drift. The two average charts read the same before/after split
(:func:`analytics.split_on_pivot`) through the pure helpers below, so they are
unit-testable without a Streamlit runtime.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from prompt_analytics import analytics
from prompt_analytics.analytics import (
    CostEngine,
    day_before,
    impact_fmt_change,
    impact_fmt_value,
)
from prompt_analytics.dashboard import data
from prompt_analytics.schema import TOKEN_TYPES

# The Compare page owns a single date widget; no cross-page persistence is needed
# (the page is self-contained), so this is a plain keyed widget.
KEY_PIVOT_DATE = "cmp_pivot_date"

# Sentinel for the "no suggestion picked" row of the suggestions selectbox.
_PICK_NONE = "— pick a date —"
_PICK_KEY = "cmp_pivot_suggest"
# Sticky guard: the last suggestion we applied to the date input, so re-selecting
# the same one (every rerun) never clobbers a date the user typed afterwards.
_PICK_APPLIED = "_cmp_pivot_suggest_applied"


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


def render_pivot_picker(frames: dict[str, pd.DataFrame]) -> str | None:
    """Switch-date picker for the Compare page; returns the chosen pivot (``YYYY-MM-DD``).

    A date input bounded to the data range, plus a selectbox of the config-change
    dates detected by :func:`analytics.suggest_pivots` (mtime of CLAUDE.md /
    settings.json) as a typing aid -- picking one fills the date, but the analysis
    never depends on it. Returns ``None`` only when there are no dates to split on.
    """
    from prompt_analytics.dashboard import filters

    bounds = filters.available_date_bounds(frames)
    if bounds is None:
        return None
    min_d, max_d = bounds

    pick_col, date_col = st.columns([3, 2])

    # Auto-suggestions: a picked one drives the date input; "pick a date" leaves
    # the user's manual choice untouched.
    directory = data.data_dir()
    suggestions = _suggested_pivots(str(directory), data._mtimes_key(directory))
    if suggestions:
        labels = {f"{day} · {label}": day for day, label in suggestions}
        with pick_col:
            choice = st.selectbox(
                "Detected config changes",
                [_PICK_NONE, *labels.keys()],
                key=_PICK_KEY,
                help="Dates a CLAUDE.md / settings.json was last modified — a likely "
                "'I changed my setup here'. Only a typing aid; the split is whatever "
                "date you set on the right.",
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
    with date_col:
        st.date_input(
            "Switch date (pivot)",
            min_value=min_d,
            max_value=max_d,
            key=KEY_PIVOT_DATE,
            help="The pivot day is counted in the AFTER side (before = up to the day before it).",
        )
    return _to_iso(st.session_state.get(KEY_PIVOT_DATE))


def render_summary_caption(report: analytics.ImpactReport, pivot: str) -> None:
    """The before/after prompt-count caption + an empty-side warning."""
    pivot_before = day_before(pivot)
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


def _render_metric_columns(metrics: list[analytics.ImpactMetric]) -> None:
    """Two columns -- Before (left) / After (right) -- one row per metric.

    The left column shows the BEFORE value; the right shows the AFTER value with
    the change as a grey delta (the same string the CLI prints). Colour is always
    ``off`` (grey): up isn't universally "good" here (cost per prompt up is bad,
    output share up is neutral), so we never imply a verdict -- the honesty ADN of
    the whole product.
    """
    before_col, after_col = st.columns(2)
    before_col.markdown("**◀ Before**")
    after_col.markdown("**After ▶**")
    for metric in metrics:
        before_col.metric(metric.label, impact_fmt_value(metric.before, metric.fmt))
        change = impact_fmt_change(metric.before, metric.after, metric.fmt)
        after_col.metric(
            metric.label,
            impact_fmt_value(metric.after, metric.fmt),
            delta=None if change == "-" else change,
            delta_color="off",
        )


def render_ratio_columns(report: analytics.ImpactReport) -> None:
    """The 5 workload-normalized ratios, Before (left) vs After (right)."""
    ratios = [m for m in report.metrics if not m.confounder]
    _render_metric_columns(ratios)


def render_confounders(report: analytics.ImpactReport) -> None:
    """The workload confounders (volume, depth, task mix) in a Before/After expander."""
    confounders = [m for m in report.metrics if m.confounder]
    if not confounders:
        return
    with st.expander("Workload confounders — how the workload itself moved (not the change)"):
        _render_metric_columns(confounders)


def render_honesty_note() -> None:
    """The standing caveat: ratios isolate the config; the split is observational."""
    st.caption(
        "Ratios are **workload-normalized** (per prompt, per turn, or as a cost share) so the "
        "change reads through the workload; the confounders describe how the workload itself "
        "moved. This is an **observational split, not a controlled experiment** — if volume, "
        "depth or task mix shifted a lot, read the deltas as correlation, not proven causation."
    )


# ---------------------------------------------------------------------------
# Pure data helpers for the two "averages" charts (no Streamlit -- unit-testable).
# ---------------------------------------------------------------------------


def token_cost_per_prompt(side: analytics.Dataset, provider: str) -> dict[str, float]:
    """Average cost **per prompt**, by token type, for one side of the split.

    An average (the per-prompt cost), never a sum: a side that simply ran more
    prompts must not look more expensive per token type. Returns ``{}`` when the
    side has no prompts; only token types with a non-zero cost are kept.
    """
    prompts = len(side.prompts)
    if not prompts:
        return {}
    engine = CostEngine(provider, side.pricing_path)
    totals: dict[str, float] = defaultdict(float)
    for row in side.tokens:
        totals[row["token_type"]] += engine.cost(
            row.get("model") or "", row["token_type"], row["token_count"]
        )
    return {tt: totals[tt] / prompts for tt in TOKEN_TYPES if totals.get(tt)}


def category_share(side: analytics.Dataset) -> dict[str, float]:
    """Share (% of prompts) per category for one side of the split.

    A mix (shares summing to 100), never counts: the two sides are compared by
    *composition*, so a bigger side does not dominate by volume. Returns ``{}``
    when the side has no prompts.
    """
    counts: Counter[str] = Counter()
    for row in side.prompts:
        category = (side.categories.get(row["prompt_id"]) or {}).get(
            "category"
        ) or "(uncategorized)"
        counts[category] += 1
    total = sum(counts.values())
    if not total:
        return {}
    return {category: 100 * n / total for category, n in counts.items()}


# The per-page before/after panel and the sidebar toggle were removed: the
# comparison now lives only on the dedicated Compare page, which composes the
# helpers above directly.
