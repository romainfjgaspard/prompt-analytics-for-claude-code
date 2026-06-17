"""Data loading and enrichment for the Streamlit dashboard.

The dashboard is a *view* over already-extracted CSVs; it does not re-implement
the joins, the schema or the pricing. It builds a single
:class:`~prompt_analytics.analytics.Dataset` from the CSVs (8.2) and reuses the
analytics layer's :class:`~prompt_analytics.analytics.CostEngine` as the one and
only place costs are computed (D3). The pandas frames below are just that
dataset reshaped for plotting, with one ``cost_<provider>_usd`` column per
pricing provider.

Data directory resolution (9.2):

* ``CCA_DEMO=1`` -> the bundled ``demo_data/`` dataset (with a banner upstream);
* else ``CCA_DATA_DIR`` -> ``PROMPT_ANALYTICS_OUTPUT_DIR`` (set by the
  ``dashboard`` command from ``--output-dir``) -> ``./output``.

Timestamps are normalized to **aware UTC** at load time (8.1): mixing tz-naive
``start_date`` with tz-aware prompt timestamps used to crash
``_available_date_bounds``.

``st.cache_data`` is keyed by a hash of the CSV mtimes (8.3) so an ``extract``
run while the dashboard is open becomes visible on the next rerun.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from prompt_analytics import analytics, schema
from prompt_analytics.config import load_config as _load_config

OUTPUT_DIR_ENV = "PROMPT_ANALYTICS_OUTPUT_DIR"
DATA_DIR_ENV = "CCA_DATA_DIR"
DEMO_ENV = "CCA_DEMO"

# CSV/file names whose mtimes invalidate the cache.
_DATA_FILES = (
    "sessions.csv",
    "prompts.csv",
    "tokens.csv",
    "token_types.csv",
    "categories.csv",
    "quota_log.csv",
    "config.yml",
)


REPO_URL = "https://github.com/romainfjgaspard/prompt-analytics-for-claude-code"
# The full recipe to get *this* dashboard on the visitor's own logs, shown in the
# "run it yourself" popover. `run --categorize` uses the local heuristic (no API
# key); `dashboard` then reads the categorized CSVs. Kept in sync with the README.
SELF_HOST_CMDS = (
    'uv tool install "prompt-analytics-for-claude-code[dashboard]" # CLI + dashboard extra, on your PATH\n'
    "prompt-analytics run --categorize # extract + snapshot + local categorize → ./output\n"
    "prompt-analytics dashboard # open the dashboard at http://localhost:8501"
)


def is_demo() -> bool:
    """True when the dashboard runs against the bundled demo dataset.

    Honors the ``CCA_DEMO`` environment variable first (how the ``dashboard``
    command and local runs set it). On Streamlit Community Cloud deploy-time
    config arrives as ``st.secrets`` rather than env vars, so fall back to a
    ``CCA_DEMO`` secret. Reading ``st.secrets`` raises when no secrets file is
    configured (the common local/test case), hence the guard.
    """
    if os.environ.get(DEMO_ENV) == "1":
        return True
    try:
        return str(st.secrets.get(DEMO_ENV)) == "1"
    except Exception:
        return False


def render_demo_banner() -> None:
    """Show the demo CTA at the top of the sidebar when on the bundled dataset (9.2).

    Every launch channel (LinkedIn / HN / Reddit) drives traffic to this demo,
    so the banner is the conversion funnel: it labels the dataset as synthetic,
    asks for a GitHub star *explicitly* (an explicit ask converts far better than
    a bare "view on GitHub" link), and shows the one-liner to run it on the
    visitor's own logs. Streamlit Community Cloud has no public app star, so the
    only star that matters -- and the metric best-of-streamlit / the gallery rank
    by -- is the GitHub one; hence the CTA points there.

    Lives in the sidebar (not the main column) so it never pushes the page's
    first chart below the fold; every page calls this *before* ``render_sidebar``
    so it lands above the filters.
    """
    if not is_demo():
        return
    bar = st.sidebar
    # Separate the CTA block from the filters above it (it's rendered at the very
    # bottom of the sidebar, right after the Category filter).
    bar.divider()
    # Lead with an explicit, always-visible invitation (the real conversion
    # goal), then the filled primary button that opens the commands. They live
    # *inside* the popover, whose panel is wider than the sidebar, so the long
    # install line isn't truncated the way an inline code block would be.
    bar.markdown(
        "**📊 Get this dashboard on your own usage** — same board, your real "
        "Claude Code data, 100% local."
    )
    with bar.popover("▶ Show me how (3 commands)", width="stretch", type="primary"):
        st.markdown("**Same dashboard, your real usage** — no API key:")
        st.code(SELF_HOST_CMDS, language="bash")
    bar.link_button("⭐ Star it on GitHub", REPO_URL, width="stretch")
    bar.caption("🎭 Demo data")
    bar.caption("_Not affiliated with Anthropic._")


# Token types whose cost is context rent: money spent re-sending context, not
# generating (the central insight of the power-user audit, 06 §1).
_RENT_TYPES = ("cache_read", "cache_write_5m", "cache_write_1h")


def _context_rent_share(tokens: pd.DataFrame, primary: str) -> float | None:
    """Share (0-100) of the primary provider's API-equivalent cost that is context rent."""
    col = cost_col(primary)
    if tokens.empty or col not in tokens.columns or "token_type" not in tokens.columns:
        return None
    total = float(tokens[col].sum())
    if total <= 0:
        return None
    rent = float(tokens.loc[tokens["token_type"].isin(_RENT_TYPES), col].sum())
    return 100.0 * rent / total


def _demo_dir() -> Path:
    """Locate the committed ``demo_data/`` directory.

    Works from an editable checkout / Streamlit Community Cloud (repo root on
    the path) and from the current working directory.
    """
    here = Path(__file__).resolve()
    candidates = [Path.cwd() / "demo_data", *(p / "demo_data" for p in here.parents)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("demo_data")


def data_dir() -> Path:
    """Resolve the directory the dashboard reads its CSVs from (9.2)."""
    if is_demo():
        override = os.environ.get(DATA_DIR_ENV)
        return Path(override) if override else _demo_dir()
    override = os.environ.get(DATA_DIR_ENV) or os.environ.get(OUTPUT_DIR_ENV)
    return Path(override) if override else Path("output")


def _mtimes_key(directory: Path) -> str:
    """A stable string of the data files' mtimes (the cache key, 8.3)."""
    parts: list[str] = []
    for name in _DATA_FILES:
        path = directory / name
        mtime = path.stat().st_mtime_ns if path.exists() else 0
        parts.append(f"{name}:{mtime}")
    return "|".join(parts)


def load_config() -> dict[str, Any]:
    """Load the merged runtime configuration for the active data directory."""
    return _load_config(data_dir())


# ---------------------------------------------------------------------------
# Cost columns (delegated to the analytics CostEngine -- no duplicated maths).
# ---------------------------------------------------------------------------


def _cost_column_name(provider: str) -> str:
    return f"cost_{provider}_usd"


def _add_cost_columns(
    tokens: pd.DataFrame, providers: list[str], pricing_path: Path | None
) -> None:
    """Add one ``cost_<provider>_usd`` column to ``tokens`` (in place).

    Vectorized over the distinct ``(model, token_type)`` pairs; the per-token
    rate itself comes from the analytics :class:`CostEngine`, so the dashboard
    and the CLI always agree on costs.
    """
    if tokens.empty:
        for provider in providers:
            tokens[_cost_column_name(provider)] = pd.Series(dtype="float64")
        return

    pairs = tokens[["model", "token_type"]].drop_duplicates().reset_index(drop=True)
    counts = pd.to_numeric(tokens["token_count"], errors="coerce").fillna(0.0)
    for provider in providers:
        engine = analytics.CostEngine(provider, pricing_path)
        # cost of one token (count=1) per pair, then scale by the real counts.
        unit = [
            engine.cost(model or "", token_type, 1)
            for model, token_type in zip(pairs["model"], pairs["token_type"], strict=True)
        ]
        rate = pairs.assign(_unit=unit)
        merged = tokens.merge(rate, on=["model", "token_type"], how="left")
        tokens[_cost_column_name(provider)] = merged["_unit"].to_numpy() * counts.to_numpy()


# ---------------------------------------------------------------------------
# Frame builders.
# ---------------------------------------------------------------------------


def _to_utc(series: pd.Series) -> pd.Series:
    """Parse to aware UTC datetimes (the 8.1 normalization)."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def _read_quota(directory: Path) -> pd.DataFrame:
    """Read ``quota_log.csv`` or an empty, correctly-typed frame."""
    path = directory / "quota_log.csv"
    if not path.exists():
        return pd.DataFrame(columns=schema.QUOTA_LOG_COLS)
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=schema.QUOTA_LOG_COLS)


def _build_frames(
    ds: analytics.Dataset, providers: list[str], directory: Path
) -> dict[str, pd.DataFrame]:
    """Reshape a :class:`Dataset` into plotting-ready pandas frames."""
    sessions = pd.DataFrame(ds.sessions, columns=schema.SESSIONS_COLS)
    prompts = pd.DataFrame(ds.prompts, columns=schema.PROMPTS_COLS)
    tokens = pd.DataFrame(ds.tokens, columns=schema.TOKENS_COLS)
    quota = _read_quota(directory)

    # --- categories -> prompts ---
    cat_rows = [
        {
            "prompt_id": pid,
            "category": info.get("category", ""),
            "complexity": info.get("complexity", ""),
        }
        for pid, info in ds.categories.items()
    ]
    categories = pd.DataFrame(cat_rows, columns=["prompt_id", "category", "complexity"])
    if not prompts.empty:
        if not categories.empty:
            prompts = prompts.merge(categories, on="prompt_id", how="left")
        else:
            prompts["category"] = pd.NA
            prompts["complexity"] = pd.NA
        for col in ("prompt_index", "char_count", "assistant_turns", "tool_calls"):
            prompts[col] = pd.to_numeric(prompts[col], errors="coerce")
        prompts["timestamp"] = _to_utc(prompts["timestamp"])

    # --- sessions ---
    if not sessions.empty:
        sessions["start_date"] = _to_utc(sessions["start_date"])

    # --- tokens: cost columns + dimensions ---
    if not tokens.empty:
        tokens["token_count"] = pd.to_numeric(tokens["token_count"], errors="coerce").fillna(0)
        tokens["token_type_label"] = tokens["token_type"].map(schema.TOKEN_TYPE_LABELS)
    _add_cost_columns(tokens, providers, ds.pricing_path)

    if not tokens.empty and not prompts.empty:
        dims = [
            c
            for c in ("prompt_id", "timestamp", "project", "category", "complexity", "prompt_index")
            if c in prompts.columns
        ]
        tokens = tokens.merge(prompts[dims], on="prompt_id", how="left")
    else:
        for col in ("timestamp", "project", "category", "complexity", "prompt_index"):
            if col not in tokens.columns:
                tokens[col] = pd.NA

    # --- derived dates (aware UTC, normalized to midnight) ---
    if "timestamp" in tokens.columns:
        tokens["date"] = _to_utc(tokens["timestamp"]).dt.normalize()
    if "timestamp" in prompts.columns:
        prompts["date"] = _to_utc(prompts["timestamp"]).dt.normalize()

    # --- per-prompt cost (sum of token rows) onto prompts ---
    cost_cols = [_cost_column_name(p) for p in providers]
    if not tokens.empty and not prompts.empty:
        per_prompt = tokens.groupby("prompt_id", as_index=False)[cost_cols].sum()
        prompts = prompts.merge(per_prompt, on="prompt_id", how="left")
        for col in cost_cols:
            prompts[col] = prompts[col].fillna(0.0)
    else:
        for col in cost_cols:
            if col not in prompts.columns:
                prompts[col] = 0.0

    # --- quota ---
    if not quota.empty:
        quota["snapshot_at"] = _to_utc(quota["snapshot_at"])
        quota["resets_at"] = _to_utc(quota["resets_at"])
        quota["utilization_pct"] = pd.to_numeric(quota["utilization_pct"], errors="coerce")

    return {
        "sessions": sessions,
        "prompts": prompts,
        "tokens": tokens,
        "quota_log": quota,
    }


@st.cache_data(show_spinner=False)
def _load_cached(data_dir_str: str, cache_key: str) -> dict[str, pd.DataFrame]:
    """Cache-keyed loader. ``cache_key`` is the CSV mtimes hash (8.3)."""
    directory = Path(data_dir_str)
    ds = analytics.dataset_from_csvs(directory)
    return _build_frames(ds, analytics.known_providers(), directory)


def providers() -> list[str]:
    """The pricing providers available, in file order (first = primary)."""
    return analytics.known_providers()


def load_all() -> dict[str, pd.DataFrame]:
    """Load, type, cost and join all frames for the active data directory."""
    directory = data_dir()
    return _load_cached(str(directory), _mtimes_key(directory))


def refresh_data() -> str:
    """Re-run extract -> snapshot -> categorize for the active data directory.

    Backs the sidebar "Refresh data" button so a user never has to drop to a
    terminal to pull in new prompts: it regenerates the CSVs the dashboard reads
    from the local ``~/.claude`` logs, snapshots the current quota, and applies
    the **heuristic** categorizer (local, no API key, no cost -- the LLM
    classifier stays a deliberate terminal action). The caller clears the
    mtime-keyed cache and reruns afterwards. Returns a one-line summary for a
    toast. Refuses to run against the bundled demo dataset (no logs to extract,
    and it must never overwrite the committed ``demo_data``).
    """
    if is_demo():
        raise RuntimeError("Refresh is disabled on the demo dataset.")
    from prompt_analytics import categorize, extract, snapshot

    directory = data_dir()
    report = extract.run_extract(directory)
    snapshot.run_snapshot(directory)
    categorize.run_categorize(output_dir=str(directory))
    return f"Extracted {report.prompts:,} prompts across {report.sessions:,} sessions."


@st.cache_data(show_spinner=False)
def _load_prompt_texts_cached(path_str: str, mtime: int) -> dict[str, str]:
    """Read ``prompts_text.csv`` into ``{prompt_id: full text}`` (empty if absent)."""
    path = Path(path_str)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if not {"prompt_id", "prompt_text"} <= set(df.columns):
        return {}
    return dict(zip(df["prompt_id"], df["prompt_text"], strict=True))


def load_prompt_texts() -> dict[str, str]:
    """Map ``prompt_id -> full prompt text`` from ``prompts_text.csv``.

    The dashboard frames carry only the truncated ``prompt_preview``; the full
    text lives in a separate file (gated by the ``prompt_text`` feature, absent
    when extracted with ``--no-text``). Returns ``{}`` when unavailable.
    """
    path = data_dir() / "prompts_text.csv"
    mtime = path.stat().st_mtime_ns if path.exists() else 0
    return _load_prompt_texts_cached(str(path), mtime)


def load_dataset() -> analytics.Dataset:
    """The raw analytics :class:`Dataset` for the active data directory.

    Used by the Session-depth page, which renders ``analytics.session_depth``
    directly rather than re-deriving the meta-analysis in pandas.
    """
    return analytics.dataset_from_csvs(data_dir())


def primary_provider() -> str:
    """The provider used for the headline cost columns (first in pricing.yml)."""
    available = providers()
    return available[0] if available else "anthropic"


def cost_col(provider: str | None = None) -> str:
    """Column name for a provider's cost (the primary provider by default)."""
    return _cost_column_name(provider or primary_provider())


def table_df(result: analytics.TableResult) -> pd.DataFrame:
    """A display-ready DataFrame from an analytics :class:`TableResult`.

    The phase-2/3 analyses (TTL, compactions, recommendations, break-even)
    return ``TableResult`` rows ready for the dashboard (7.4); this maps the
    row keys to the human column labels, preserving the column order.
    """
    df = pd.DataFrame(result.rows, columns=[c.key for c in result.columns])
    renamed: pd.DataFrame = df.rename(columns={c.key: c.label for c in result.columns})
    return renamed


def dominant_model_per_session(prompts: pd.DataFrame) -> pd.DataFrame:
    """``[session_id, model]`` using each session's most frequent model.

    Shared by the Models and Sessions pages (was copy-pasted, A4). Ties are
    broken by model name so the result is deterministic.
    """
    if prompts.empty or not {"session_id", "model"} <= set(prompts.columns):
        return pd.DataFrame(columns=["session_id", "model"])
    work = prompts.dropna(subset=["session_id", "model"])
    if work.empty:
        return pd.DataFrame(columns=["session_id", "model"])
    counts = work.groupby(["session_id", "model"]).size().reset_index(name="n")
    counts = counts.sort_values(["session_id", "n", "model"], ascending=[True, False, True])
    result: pd.DataFrame = counts.drop_duplicates("session_id")[["session_id", "model"]]
    return result


def box_stats(values: Iterable[float], *, lo: float = 0.05, hi: float = 0.95) -> list[float]:
    """ECharts boxplot 5-number summary ``[w_lo, Q1, median, Q3, w_hi]``.

    The whiskers are the ``lo``/``hi`` percentiles (default **p5/p95**) instead of
    the raw min/max, so a long tail of extreme values does not stretch the y-range
    and flatten every box. Pair with :func:`box_cap` to clip the axis just above
    the tallest whisker and count the observations that spill over (the honest way
    to keep the boxes legible without hiding that outliers exist). Lives here, not
    in ``echarts.py``, so it stays unit-testable (``echarts`` can't be imported
    outside a real Streamlit server).
    """
    arr = np.asarray(list(values), dtype=float)
    q = np.quantile(arr, [lo, 0.25, 0.5, 0.75, hi])
    return [round(float(v), 4) for v in q]


def box_cap(
    groups: Iterable[Iterable[float]], stats: list[list[float]]
) -> tuple[float | None, int]:
    """A y-axis ``max`` just above the tallest **box** (Q3), plus the count clipped.

    Capping above the tallest *whisker* is useless when a few categories have a
    genuinely high p95 -- that whisker is exactly what stretches the axis and
    flattens every box. Instead we cap 25% above the tallest Q3, so every box
    fills real vertical space and the taller whiskers / outliers clip off the top.
    ``stats`` are the per-group :func:`box_stats` (``[w_lo, Q1, median, Q3, w_hi]``).
    Returns ``(y_max, n_above)`` where ``n_above`` is how many raw observations
    across all ``groups`` exceed ``y_max`` -- surface it in a caption so the clip
    stays honest. Returns ``(None, 0)`` when there is nothing to clip.
    """
    cap = max((s[-2] for s in stats), default=0.0)  # tallest Q3 (box top)
    if cap <= 0:
        return None, 0
    y_max = round(cap * 1.25, 2)
    above = sum(int((np.asarray(list(g), dtype=float) > y_max).sum()) for g in groups)
    return y_max, int(above)


# Time-series granularity (8.x UX): a single daily bar is useless over months,
# so charts group by Day / Week / Month, defaulting to a sensible grain for the
# observed span. Shared so the app and Models pages bucket identically.
GRANULARITIES = ("Day", "Week", "Month")


def auto_granularity(dates: Iterable[Any]) -> str:
    """Default grain for a date span: Day (<~1mo), Week (<~6mo), else Month."""
    ds = pd.to_datetime(pd.Series(list(dates)), utc=True, errors="coerce").dropna()
    if ds.empty:
        return "Day"
    span_days = int((ds.max() - ds.min()).days)
    if span_days <= 31:
        return "Day"
    if span_days <= 183:
        return "Week"
    return "Month"


def to_period(dates: pd.Series, granularity: str) -> pd.Series:
    """Floor each timestamp to the start of its Day / Week (Monday) / Month bucket."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    if granularity == "Week":
        out = pd.DatetimeIndex(idx - pd.to_timedelta(idx.weekday, unit="D")).normalize()
    elif granularity == "Month":
        out = idx.to_period("M").to_timestamp().tz_localize("UTC")
    else:
        out = idx.normalize()
    return pd.Series(out, index=dates.index)


def period_label(ts: Any, granularity: str) -> str:
    """Axis label for a period-start timestamp (``2026-06`` for months, else ISO date)."""
    t = pd.Timestamp(ts)
    return t.strftime("%Y-%m") if granularity == "Month" else t.strftime("%Y-%m-%d")
