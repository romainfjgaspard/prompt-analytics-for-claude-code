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
from collections import defaultdict
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

# Comparison windows: how much of each side of the pivot to keep. ``1 week`` /
# ``1 month`` keep N calendar days *immediately* on each side (a like-for-like
# window for measuring the effect of something installed on the pivot day); ``Full
# history`` (None) keeps everything -- the only mode whose ratios match the CLI
# ``impact`` table, which always runs on the whole history.
KEY_WINDOW = "cmp_window"
WINDOW_DAYS: dict[str, int | None] = {
    "1 week": 7,
    "1 month": 30,
    "Full history": None,
}


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


def render_pivot_picker(frames: dict[str, pd.DataFrame]) -> tuple[str | None, int | None]:
    """The Compare page's top controls, on a single row; returns ``(pivot, window_days)``.

    Three controls side by side -- the detected-config-change selectbox (a typing
    aid: picking one fills the date, but the analysis never depends on it), the
    switch date bounded to the data range, and the comparison window. ``pivot`` is
    ``YYYY-MM-DD`` (``None`` only when there are no dates to split on); ``window_days``
    is the per-side window in days, or ``None`` for the full history.
    """
    from prompt_analytics.dashboard import filters

    bounds = filters.available_date_bounds(frames)
    if bounds is None:
        return None, None
    min_d, max_d = bounds

    pick_col, date_col, win_col = st.columns([4, 3, 3])

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
                "date you set next to it.",
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
    with win_col:
        window_label = st.radio(
            "Comparison window",
            list(WINDOW_DAYS),
            index=len(WINDOW_DAYS) - 1,  # default: Full history (matches the CLI)
            horizontal=True,
            key=KEY_WINDOW,
            help="Restrict each side to N days right around the pivot for a fair like-for-like "
            "read (1 week = the 7 days before vs the 7 after), or keep the whole history.",
        )

    return _to_iso(st.session_state.get(KEY_PIVOT_DATE)), WINDOW_DAYS[window_label]


def window_dataset(ds: analytics.Dataset, pivot: str, window_days: int | None) -> analytics.Dataset:
    """Restrict ``ds`` to ``window_days`` on each side of ``pivot`` (else the whole set).

    For ``window_days=N`` the kept range is ``[pivot - N, pivot + N - 1]`` so that,
    once :func:`analytics.split_on_pivot` cuts it on the pivot, *before* covers the
    N days up to the day before the pivot and *after* covers the pivot day plus the
    next N-1 -- two equal-length windows for a fair like-for-like comparison. A
    ``None`` window returns ``ds`` unchanged (the full history, matching the CLI).
    """
    if window_days is None:
        return ds
    day = _dt.date.fromisoformat(pivot)
    since = (day - _dt.timedelta(days=window_days)).isoformat()
    until = (day + _dt.timedelta(days=window_days - 1)).isoformat()
    return analytics.filter_dates(ds, since, until)


# ---------------------------------------------------------------------------
# Composition shift: prose/code and context load/rent shares, before vs after.
# Pure (no Streamlit) so they are unit-testable; ``None`` when a side is empty.
# ---------------------------------------------------------------------------


def output_code_share(comp: analytics.OutputComposition) -> float | None:
    """Code's share (%) of the generated output spend, or ``None`` with no output."""
    total = comp.prose_cost + comp.code_cost
    return 100 * comp.code_cost / total if total else None


def output_test_share(comp: analytics.OutputComposition) -> float | None:
    """Tests' share (%) of the code lines written, or ``None`` with no lines added."""
    return 100 * comp.total_test / comp.total_added if comp.total_added else None


def context_rent_share(ctx: analytics.ContextCost) -> float | None:
    """Rent's share (%) of the context cost, or ``None`` with no context cost."""
    return 100 * ctx.rent_cost / ctx.total_cost if ctx.total_cost else None


def output_cost_per_line(comp: analytics.OutputComposition) -> float | None:
    """Code generation cost ($) per line of code written, or ``None`` with no lines.

    The dollars spent on the *code* half of the output, per line actually added --
    an efficiency ratio that drops when the same code costs less to generate (e.g.
    a cheaper model), independently of how much was written.
    """
    return comp.code_cost / comp.total_added if comp.total_added else None


def output_tokens_per_line(comp: analytics.OutputComposition) -> float | None:
    """Generated code tokens per line of code written, or ``None`` with no lines.

    The token cost of a line of code regardless of model pricing -- the twin of
    :func:`output_cost_per_line` that isolates *verbosity* from the price per token,
    so a model switch and a terser-output change read apart.
    """
    return comp.code_tokens / comp.total_added if comp.total_added else None


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
    """Before then After, each as one horizontal band of metric cards.

    A **Before** row of cards over an **After** row of the same cards: each metric
    sits in the same column on both rows, so the pair reads top-to-bottom and the
    grid stays dense (two bands, not one tall row per metric). The After card
    carries the change as a grey delta (the same string the CLI prints). Colour is
    always ``off`` (grey): up isn't universally "good" here (cost per prompt up is
    bad, output share up is neutral), so we never imply a verdict -- the honesty ADN
    of the whole product.
    """
    if not metrics:
        return
    st.markdown("**◀ Before**")
    for col, metric in zip(st.columns(len(metrics)), metrics, strict=True):
        col.metric(metric.label, impact_fmt_value(metric.before, metric.fmt))
    st.markdown("**After ▶**")
    for col, metric in zip(st.columns(len(metrics)), metrics, strict=True):
        change = impact_fmt_change(metric.before, metric.after, metric.fmt)
        col.metric(
            metric.label,
            impact_fmt_value(metric.after, metric.fmt),
            delta=None if change == "-" else change,
            delta_color="off",
        )


def render_share_columns(rows: list[tuple[str, float | None, float | None, str]]) -> None:
    """Aligned Before/After cards for a list of ``(label, before, after, fmt)`` rows.

    A thin wrapper over :func:`_render_metric_columns` for the composition-shift
    cards (shares plus the per-line efficiency metrics): each row carries its own
    ``fmt`` (``pct`` / ``money`` / ``ratio``), so the delta reads in the right unit
    -- the same honest, verdict-free badge as the headline ratios.
    """
    metrics = [
        analytics.ImpactMetric(label, before, after, fmt) for label, before, after, fmt in rows
    ]
    _render_metric_columns(metrics)


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


def language_share(side: analytics.Dataset, provider: str) -> dict[str, float]:
    """Share (% of code lines written) per language for one side of the split.

    A mix (shares summing to 100), never counts: the two sides are compared by the
    *composition* of the output, so a side that simply wrote more code does not
    dominate by volume. Reads :func:`analytics.output_composition` (the same source
    the Composition tab charts), so the language split here matches that tab.
    Returns ``{}`` when no code lines were written on the side.
    """
    comp = analytics.output_composition(side, provider)
    lines = {lng.language: lng.lines_added for lng in comp.languages if lng.lines_added > 0}
    total = sum(lines.values())
    if not total:
        return {}
    return {language: 100 * n / total for language, n in lines.items()}


# The per-page before/after panel and the sidebar toggle were removed: the
# comparison now lives only on the dedicated Compare page, which composes the
# helpers above directly.
