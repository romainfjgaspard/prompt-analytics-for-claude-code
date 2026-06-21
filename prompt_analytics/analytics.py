"""Pure-Python analytics layer: joins, aggregations and read-time costs.

This is the single place where tokens x prompts x sessions x categories are
joined and priced (D3: ``tokens.csv`` stores raw counts; costs are always
computed here, at read time, from the pricing tables). It is consumed by the
CLI commands and (phase 8) by the dashboard -- it must therefore stay free of
streamlit and pandas; the volumes (thousands of prompts) make stdlib plenty.

On-the-fly mode (7.2): :func:`load_dataset` works without a prior ``extract``.
When the ``output/`` CSVs exist and are fresher than every JSONL file under
``~/.claude/projects``, they are used as a cache; otherwise the history is
parsed in memory (fast, thanks to the per-file parse cache).

Cost notes:

* ``server_tool_use`` counts server-side *requests*, not tokens; the pricing
  grids are per-token, so those rows are excluded from costs and from token
  totals (they are still shown in ``summary``).
* Models without a pricing entry contribute 0 to costs and are surfaced in
  the result notes (never silently).
"""

from __future__ import annotations

import contextlib
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import extract, paths
from .compose import detect_kind
from .context import NO_LANGUAGE, NO_PATH
from .pricing import get_model_pricing, get_per_request, is_long_context, load_pricing
from .schema import (
    REQUESTS_COLS,
    TOKEN_TYPE_LABELS,
    TOKEN_TYPES,
    TOKENS_COLS,
    UNATTRIBUTED_SOURCE,
)

__all__ = [
    "Column",
    "TableResult",
    "Dataset",
    "CostEngine",
    "load_dataset",
    "dataset_from_csvs",
    "filter_project",
    "filter_dates",
    "filter_prompt_ids",
    "split_on_pivot",
    "known_providers",
    "summary",
    "by_project",
    "by_model",
    "by_token_type",
    "by_category",
    "input_cost",
    "by_output",
    "output_composition",
    "OutputComposition",
    "LanguageComposition",
    "by_context",
    "context_cost",
    "ContextCost",
    "ContextElementCost",
    "by_task",
    "task_graph",
    "TaskGraph",
    "TaskNode",
    "TaskSatellite",
    "top_prompts",
    "sessions_table",
    "session_depth",
    "context_growth",
    "ttl_losses",
    "compactions",
    "session_overhead",
    "model_category",
    "recommendations",
    "burn_rate",
    "timeline",
    "break_even",
    "compare_providers",
    "impact",
    "impact_report",
    "ImpactReport",
    "ImpactMetric",
    "suggest_pivots",
    "day_before",
    "impact_fmt_value",
    "impact_fmt_change",
    "flat_export",
    "mini_summary",
]

# Token types that are priced per token. server_tool_use is billed per
# request (no per-request price in the grids), so it never enters costs.
COSTED_TOKEN_TYPES = frozenset(
    {"input", "output", "cache_read", "cache_write_5m", "cache_write_1h"}
)

# Pseudo model id used by Claude Code for synthetic (non-API) messages.
_SYNTHETIC_MODEL = "<synthetic>"

# Integer columns of requests.csv (everything but the ids, timestamp, model
# and stop_reason).
_REQUEST_INT_COLS = tuple(
    col
    for col in REQUESTS_COLS
    if col not in ("session_id", "prompt_id", "timestamp", "model", "stop_reason")
)

# Depth bands for the session-depth meta-analysis (lo, hi, label); hi=None
# means unbounded.
DEPTH_BANDS: tuple[tuple[int, int | None, str], ...] = (
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, 5, "5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 50, "21-50"),
    (51, None, "51+"),
)


def _depth_band(index: int) -> str:
    """The DEPTH_BANDS label a 1-based prompt index falls into."""
    for lo, hi, label in DEPTH_BANDS:
        if index >= lo and (hi is None or index <= hi):
            return label
    return DEPTH_BANDS[-1][2]


# ---------------------------------------------------------------------------
# Result shapes shared with the renderer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One column of a :class:`TableResult`.

    ``kind`` drives display formatting only (raw values stay in the rows for
    ``--format csv|json``): ``str``, ``int``, ``money``, ``pct``, ``x``
    (multiplier), ``num``.
    """

    key: str
    label: str
    kind: str = "str"


@dataclass
class TableResult:
    """A renderable tabular result (table / csv / json agnostic)."""

    title: str
    columns: list[Column]
    rows: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset loading (7.2: on-the-fly with output/ as a fresh cache).
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """All extraction rows plus categorization, ready to aggregate.

    ``requests`` is the request grain (one row per deduplicated API request,
    V10): the substrate of the TTL / compaction / accumulated-context
    analyses (phase 2). Empty when reading a pre-v2 extract directory.
    """

    sessions: list[dict[str, Any]]
    prompts: list[dict[str, Any]]
    tokens: list[dict[str, Any]]
    categories: dict[str, dict[str, str]]
    source: str  # human-readable provenance for the notes
    pricing_path: Path | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    # Output composition (Axe C). Long per (prompt_id, language, kind); and the
    # per-prompt prose/code split of the generated output tokens. Empty when
    # reading a pre-Axe-C extract (the `by-output` view degrades gracefully).
    output_files: list[dict[str, Any]] = field(default_factory=list)
    output_tokens: list[dict[str, Any]] = field(default_factory=list)
    # Context composition (Axe D, static snapshot). Long per
    # (session_id, source, language): the local-tokenizer size of each context
    # source. Empty when reading a pre-Axe-D extract.
    context_sources: list[dict[str, Any]] = field(default_factory=list)
    # Context cost over time (Axe D, D2). Long per (session, source, language,
    # model): the real cache tokens attributed by size x turns of presence (rent)
    # and one-off loading (write). Empty when reading a pre-D2 extract.
    context_cost: list[dict[str, Any]] = field(default_factory=list)
    # Task attribution (Axe B2). ``tasks`` is the task dimension (one row per
    # task); ``task_prompts`` the prompt->task membership edges. A task's cost is
    # derived at read time by joining ``task_prompts`` -> ``tokens``. Empty when
    # reading a pre-B2 extract (the `by-task` view degrades gracefully).
    tasks: list[dict[str, Any]] = field(default_factory=list)
    task_prompts: list[dict[str, Any]] = field(default_factory=list)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _coerce_int(rows: list[dict[str, Any]], cols: tuple[str, ...]) -> None:
    for row in rows:
        for col in cols:
            try:
                row[col] = int(row.get(col) or 0)
            except (TypeError, ValueError):
                row[col] = 0


def _csvs_fresh(output_dir: Path, claude_dir: Path | None) -> bool:
    """True when the output CSVs exist, carry the current schema, and are
    newer than every JSONL file of the history."""
    paths = [
        output_dir / name for name in ("sessions.csv", "prompts.csv", "tokens.csv", "requests.csv")
    ]
    if not all(p.exists() for p in paths):
        return False
    # Schema canaries: pre-phase-7 tokens.csv has no model column, and a
    # pre-v2 extract has no requests.csv at all (checked above) -- the
    # request-grain analyses must never run on a silently empty table.
    for path, expected in ((paths[2], TOKENS_COLS), (paths[3], REQUESTS_COLS)):
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != expected:
            return False
    csv_mtime = min(p.stat().st_mtime_ns for p in paths)
    jsonl_files = extract._iter_jsonl_files(claude_dir)
    if not jsonl_files:
        return True
    return max(f.stat().st_mtime_ns for f in jsonl_files) <= csv_mtime


def _window_label(data_dir: Path) -> str:
    """Human form of the persisted extract window, or "" (1.5).

    ``extract --since/--until`` writes ``extract_meta.json``; any reader of
    those CSVs must say the cache is partial instead of serving it as the
    full history.
    """
    try:
        meta = json.loads((data_dir / "extract_meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    window = meta.get("window") if isinstance(meta, dict) else None
    if not isinstance(window, dict):
        return ""
    parts = [f"{bound} {window[bound]}" for bound in ("since", "until") if window.get(bound)]
    return "window: " + ", ".join(parts) if parts else ""


def _load_categories(path: Path) -> dict[str, dict[str, str]]:
    """``prompt_id -> {category, complexity}`` from categories.csv (if any)."""
    if not path.exists():
        return {}
    categories: dict[str, dict[str, str]] = {}
    for row in _read_csv_rows(path):
        pid = row.get("prompt_id", "")
        if pid and (row.get("category") or row.get("complexity")):
            categories[pid] = {
                "category": row.get("category", ""),
                "complexity": row.get("complexity", ""),
            }
    return categories


def load_dataset(
    output_dir: Path,
    *,
    claude_dir: Path | None = None,
    use_cache: bool = True,
    pricing_path: Path | None = None,
) -> Dataset:
    """Load the dataset, from the output CSVs when fresh, else live.

    Args:
        output_dir: Directory holding (or not) the extraction CSVs.
        claude_dir: Override of ``~/.claude/projects`` (mainly for tests).
        use_cache: When False, bypass both the CSV cache and the per-file
            parse cache (always re-parse from the JSONL files).
        pricing_path: Optional custom pricing YAML, carried on the dataset.

    Returns:
        The loaded :class:`Dataset`. ``categories.csv`` is always read from
        ``output_dir`` when present (it is authored by ``categorize``, not
        regenerable from the JSONL history).
    """
    output_dir = Path(output_dir)
    if use_cache and _csvs_fresh(output_dir, claude_dir):
        window = _window_label(output_dir)
        fresh = f"fresh, {window}" if window else "fresh"
        return dataset_from_csvs(
            output_dir, pricing_path=pricing_path, source=f"{output_dir} CSVs ({fresh})"
        )
    source = f"live parse of {paths.claude_projects_display()}"
    # Be loud when CSVs exist but are older than the JSONL history: the user
    # may believe they are analyzing that export when they are not.
    if use_cache and all(
        (output_dir / name).exists() for name in ("sessions.csv", "prompts.csv", "tokens.csv")
    ):
        source += (
            f" -- stale CSVs in {output_dir} IGNORED"
            " (re-run `prompt-analytics extract` to refresh them)"
        )
    # no_text=False keeps the prompt previews: a live parse writes NOTHING to
    # disk, so privacy is unaffected, and `prompts --top` / `export --flat`
    # stay actionable even when the CSV cache is not fresh (D1).
    result = extract.collect(no_text=False, use_cache=use_cache, claude_dir=claude_dir)
    return Dataset(
        sessions=[dict(row) for row in result.sessions],
        prompts=[dict(row) for row in result.prompts],
        tokens=[dict(row) for row in result.tokens],
        categories=_load_categories(output_dir / "categories.csv"),
        source=source,
        pricing_path=pricing_path,
        requests=[dict(row) for row in result.requests],
        output_files=[dict(row) for row in result.output_files],
        output_tokens=[dict(row) for row in result.output_tokens],
        context_sources=[dict(row) for row in result.context_sources],
        context_cost=[dict(row) for row in result.context_cost],
        tasks=[dict(row) for row in result.tasks],
        task_prompts=[dict(row) for row in result.task_prompts],
    )


def dataset_from_csvs(
    data_dir: Path,
    *,
    pricing_path: Path | None = None,
    source: str | None = None,
) -> Dataset:
    """Build a :class:`Dataset` by reading the CSVs in ``data_dir`` directly.

    Unlike :func:`load_dataset`, this never falls back to a live parse and never
    compares mtimes against ``~/.claude/projects``: it simply visualizes the
    CSVs it is given. This is what the dashboard (and the demo dataset) need --
    they render already-extracted data, not the local Claude Code history.

    Missing CSVs yield empty sections (the caller decides what to do with an
    empty dataset). ``categories.csv`` is joined when present.
    """
    data_dir = Path(data_dir)

    def _rows(name: str) -> list[dict[str, Any]]:
        path = data_dir / name
        return _read_csv_rows(path) if path.exists() else []

    sessions = _rows("sessions.csv")
    prompts = _rows("prompts.csv")
    tokens = _rows("tokens.csv")
    requests = _rows("requests.csv")
    output_files = _rows("output_files.csv")
    output_tokens = _rows("output_tokens.csv")
    context_sources = _rows("context_sources.csv")
    context_cost = _rows("context_cost.csv")
    tasks = _rows("tasks.csv")
    task_prompts = _rows("task_prompts.csv")
    _coerce_int(prompts, ("prompt_index", "char_count", "assistant_turns", "tool_calls"))
    _coerce_int(tokens, ("token_count", "is_sidechain"))
    _coerce_int(requests, _REQUEST_INT_COLS)
    _coerce_int(output_files, ("edits", "lines_added", "lines_deleted"))
    _coerce_int(output_tokens, ("output_prose_tokens", "output_code_tokens"))
    _coerce_int(context_sources, ("tokens", "items"))
    _coerce_int(
        context_cost,
        ("rent_read_tokens", "load_write_5m_tokens", "load_write_1h_tokens"),
    )
    _coerce_int(tasks, ("prompts",))
    if source is None:
        window = _window_label(data_dir)
        source = f"{data_dir} CSVs ({window})" if window else f"{data_dir} CSVs"
    return Dataset(
        sessions=sessions,
        prompts=prompts,
        tokens=tokens,
        categories=_load_categories(data_dir / "categories.csv"),
        source=source,
        pricing_path=pricing_path,
        requests=requests,
        output_files=output_files,
        output_tokens=output_tokens,
        context_sources=context_sources,
        context_cost=context_cost,
        tasks=tasks,
        task_prompts=task_prompts,
    )


def filter_project(ds: Dataset, project: str) -> Dataset:
    """A view of ``ds`` restricted to one project (2.8: ``sessions --project``).

    Sessions are kept by their ``project``; prompts, tokens and requests follow
    by ``session_id`` so the per-session costs stay coherent (a prompt's project
    normally equals its session's). ``categories``/``source``/``pricing_path``
    ride along unchanged.
    """
    session_ids = {
        row["session_id"] for row in ds.sessions if (row.get("project") or "") == project
    }
    kept_prompts = [row for row in ds.prompts if row["session_id"] in session_ids]
    kept_prompt_ids = {row["prompt_id"] for row in kept_prompts}
    return Dataset(
        sessions=[row for row in ds.sessions if row["session_id"] in session_ids],
        prompts=kept_prompts,
        tokens=[row for row in ds.tokens if row["session_id"] in session_ids],
        categories=ds.categories,
        source=ds.source,
        pricing_path=ds.pricing_path,
        requests=[row for row in ds.requests if row["session_id"] in session_ids],
        output_files=[row for row in ds.output_files if row["prompt_id"] in kept_prompt_ids],
        output_tokens=[row for row in ds.output_tokens if row["prompt_id"] in kept_prompt_ids],
        context_sources=[row for row in ds.context_sources if row["session_id"] in session_ids],
        context_cost=[row for row in ds.context_cost if row["session_id"] in session_ids],
        tasks=[row for row in ds.tasks if row["session_id"] in session_ids],
        task_prompts=[row for row in ds.task_prompts if row["prompt_id"] in kept_prompt_ids],
    )


def filter_dates(ds: Dataset, since: str | None, until: str | None) -> Dataset:
    """A view of ``ds`` restricted to prompts dated within ``[since, until]``.

    ``since``/``until`` are inclusive ``YYYY-MM-DD`` strings (either may be
    None). A real prompt is kept when the calendar day of its stored
    ``timestamp`` falls in the range -- a lexical compare on the ``YYYY-MM-DD``
    prefix, the same day convention as ``burn-rate``/``timeline`` (no timezone
    re-interpretation at read time). Tokens and requests follow their prompt by
    ``prompt_id``; pseudo-prompt rows (continuations, with no prompts.csv row
    and no timestamp) ride along when their session keeps at least one prompt.
    Sessions left with no kept prompt are dropped.
    """
    if not since and not until:
        return ds

    def _in_range(day: str) -> bool:
        if not day:
            return False
        if since and day < since:
            return False
        return not (until and day > until)

    kept_prompts = [row for row in ds.prompts if _in_range(_parse_day(row.get("timestamp", "")))]
    kept_prompt_ids = {row["prompt_id"] for row in kept_prompts}
    kept_session_ids = {row["session_id"] for row in kept_prompts}
    real_prompt_ids = _real_prompt_ids(ds)

    def _keep_usage(row: dict[str, Any]) -> bool:
        pid = row.get("prompt_id", "")
        if pid in real_prompt_ids:
            return pid in kept_prompt_ids
        # Pseudo-prompt (continuation) rows have no timestamp: keep them with
        # their session so per-session costs stay coherent.
        return row.get("session_id", "") in kept_session_ids

    return Dataset(
        sessions=[row for row in ds.sessions if row["session_id"] in kept_session_ids],
        prompts=kept_prompts,
        tokens=[row for row in ds.tokens if _keep_usage(row)],
        categories=ds.categories,
        source=ds.source,
        pricing_path=ds.pricing_path,
        requests=[row for row in ds.requests if _keep_usage(row)],
        # Output rows exist for real prompts only -> follow their prompt_id.
        output_files=[row for row in ds.output_files if row["prompt_id"] in kept_prompt_ids],
        output_tokens=[row for row in ds.output_tokens if row["prompt_id"] in kept_prompt_ids],
        # Context rows are per session -> keep those whose session keeps a prompt.
        context_sources=[
            row for row in ds.context_sources if row["session_id"] in kept_session_ids
        ],
        context_cost=[row for row in ds.context_cost if row["session_id"] in kept_session_ids],
        # Tasks follow their session; membership edges follow their kept prompt.
        tasks=[row for row in ds.tasks if row["session_id"] in kept_session_ids],
        task_prompts=[row for row in ds.task_prompts if row["prompt_id"] in kept_prompt_ids],
    )


def filter_prompt_ids(ds: Dataset, prompt_ids: set[str] | frozenset[str]) -> Dataset:
    """A view of ``ds`` restricted to a set of prompt ids (dashboard cross-filter).

    The dashboard applies its sidebar / chart-click selection on the pandas
    frames, then hands the surviving prompt ids here so the output-composition
    view honours the very same filter as every other tab. The Axe-C analyses
    read only prompts / tokens / output rows, so sessions and requests ride
    along unnarrowed (cheaper, and they are not consulted).
    """
    kept = set(prompt_ids)
    return Dataset(
        sessions=ds.sessions,
        prompts=[row for row in ds.prompts if row.get("prompt_id") in kept],
        tokens=[row for row in ds.tokens if row.get("prompt_id") in kept],
        categories=ds.categories,
        source=ds.source,
        pricing_path=ds.pricing_path,
        requests=ds.requests,
        output_files=[row for row in ds.output_files if row.get("prompt_id") in kept],
        output_tokens=[row for row in ds.output_tokens if row.get("prompt_id") in kept],
        # Session-grain context rows ride along unnarrowed, like sessions.
        context_sources=ds.context_sources,
        context_cost=ds.context_cost,
        # The task dimension rides along; membership edges honour the selection.
        tasks=ds.tasks,
        task_prompts=[row for row in ds.task_prompts if row.get("prompt_id") in kept],
    )


def known_providers(pricing_path: Path | None = None) -> list[str]:
    """The provider keys available in the pricing file, in file order."""
    return list(load_pricing(pricing_path).get("providers", {}))


# ---------------------------------------------------------------------------
# Cost engine: raw counts + pricing -> USD, loud about unpriced models.
# ---------------------------------------------------------------------------


class CostEngine:
    """Prices raw token counts for one provider, tracking unpriced models."""

    def __init__(self, provider: str, pricing_path: Path | None = None) -> None:
        self.provider = provider
        self.pricing_path = pricing_path
        self.unpriced: set[str] = set()
        self.long_context: set[str] = set()
        self._rates: dict[str, dict[str, Any] | None] = {}
        self._per_request: dict[str, float | None] = {}

    def _rate(self, model: str) -> dict[str, Any] | None:
        if model not in self._rates:
            self._rates[model] = get_model_pricing(model, self.provider, self.pricing_path)
        return self._rates[model]

    def _per_request_rate(self, key: str) -> float | None:
        if key not in self._per_request:
            self._per_request[key] = get_per_request(key, self.provider, self.pricing_path)
        return self._per_request[key]

    def cost(self, model: str, token_type: str, count: int) -> float:
        """USD cost of ``count`` units of ``token_type`` on ``model`` (0 if unpriced).

        Per-token types use the model's grid; ``server_tool_use`` counts
        *requests* and is billed from the provider's ``per_request`` table (3.3,
        0 when none is configured). A stripped long-context suffix (3.2) is
        tracked so it can be flagged: the tier is billed at the base rate.
        """
        if not count:
            return 0.0
        if token_type == "server_tool_use":
            rate = self._per_request_rate("server_tool_use")
            return count * rate if rate is not None else 0.0
        if token_type not in COSTED_TOKEN_TYPES:
            return 0.0
        rates = self._rate(model)
        if rates is None:
            if model and model != _SYNTHETIC_MODEL:
                self.unpriced.add(model)
            return 0.0
        if model and is_long_context(model):
            self.long_context.add(model)
        return count * float(rates[token_type]) / 1_000_000

    def note(self) -> str | None:
        """A warning line listing unpriced models, or None."""
        if not self.unpriced:
            return None
        return (
            f"WARNING: model(s) without {self.provider} pricing (counted at $0): "
            + ", ".join(sorted(self.unpriced))
            + " -- add an entry or a prefix fallback to pricing.yml"
        )

    def long_context_note(self) -> str | None:
        """A warning that long-context usage was priced at the base rate (3.2), or None."""
        if not self.long_context:
            return None
        return (
            "NOTE: long context priced at base rate (no >200K premium modelled) for: "
            + ", ".join(sorted(self.long_context))
            + " -- current Claude models bill the 1M window at the base rate; "
            "older models had a >200K premium not in this grid."
        )


# ---------------------------------------------------------------------------
# Small shared joins.
# ---------------------------------------------------------------------------


def _real_prompt_ids(ds: Dataset) -> set[str]:
    return {row["prompt_id"] for row in ds.prompts}


def _project_of(ds: Dataset) -> dict[str, str]:
    """``prompt_id -> project`` (real prompts), used to group token rows."""
    return {row["prompt_id"]: row.get("project") or "" for row in ds.prompts}


def _session_project(ds: Dataset) -> dict[str, str]:
    return {row["session_id"]: row.get("project") or "" for row in ds.sessions}


def _token_total(counts: Counter[str]) -> int:
    """Total tokens, excluding per-request server_tool_use counts."""
    return sum(count for token_type, count in counts.items() if token_type != "server_tool_use")


def _prompt_costs(ds: Dataset, engine: CostEngine) -> dict[str, float]:
    """``prompt_id -> USD`` over ALL token rows (pseudo-prompts included)."""
    costs: dict[str, float] = defaultdict(float)
    for row in ds.tokens:
        costs[row["prompt_id"]] += engine.cost(
            row.get("model") or "", row["token_type"], row["token_count"]
        )
    return costs


def _prompt_token_counts(ds: Dataset) -> dict[str, Counter[str]]:
    """``prompt_id -> Counter(token_type -> count)`` (models merged)."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ds.tokens:
        counts[row["prompt_id"]][row["token_type"]] += row["token_count"]
    return counts


def _source_note(ds: Dataset) -> str:
    return f"Source: {ds.source}."


def _parse_day(value: str) -> str:
    """The YYYY-MM-DD part of an ISO timestamp (empty-safe)."""
    return value[:10] if value else ""


# ---------------------------------------------------------------------------
# Commands (7.3).
# ---------------------------------------------------------------------------


def summary(ds: Dataset, providers: list[str] | None = None) -> TableResult:
    """Overview: sessions, prompts, period, tokens by type, cost per provider."""
    providers = providers or known_providers(ds.pricing_path)
    rows: list[dict[str, Any]] = [
        {"metric": "Sessions", "value": len(ds.sessions)},
        {"metric": "Prompts", "value": len(ds.prompts)},
        {"metric": "Projects", "value": len({p.get("project") or "" for p in ds.prompts})},
    ]
    days = sorted(_parse_day(p.get("timestamp", "")) for p in ds.prompts if p.get("timestamp"))
    if days:
        first, last = days[0], days[-1]
        span = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days + 1
        rows.append({"metric": "Period", "value": f"{first} .. {last} ({span} days)"})

    totals: Counter[str] = Counter()
    for row in ds.tokens:
        totals[row["token_type"]] += row["token_count"]
    for token_type in TOKEN_TYPES:
        if token_type == "server_tool_use" and not totals.get(token_type):
            continue
        label = TOKEN_TYPE_LABELS[token_type]
        if token_type == "server_tool_use":
            label += " (requests)"
        else:
            label += " tokens"
        rows.append({"metric": label, "value": f"{totals.get(token_type, 0):,}"})
    rows.append({"metric": "Total tokens", "value": f"{_token_total(totals):,}"})

    notes = [_source_note(ds)]
    for provider in providers:
        engine = CostEngine(provider, ds.pricing_path)
        total = sum(_prompt_costs(ds, engine).values())
        rows.append({"metric": f"Cost ({provider})", "value": f"${total:,.2f}"})
        if (note := engine.note()) is not None:
            notes.append(note)
        if (lc_note := engine.long_context_note()) is not None:
            notes.append(lc_note)

    # Subagents first-class (1.2): what the sidechains cost, as a share of
    # the bill (priced on the first provider).
    if providers and any(row.get("is_sidechain") for row in ds.tokens):
        engine = CostEngine(providers[0], ds.pricing_path)
        subagent_cost = sum(
            engine.cost(row.get("model") or "", row["token_type"], row["token_count"])
            for row in ds.tokens
            if row.get("is_sidechain")
        )
        total = sum(_prompt_costs(ds, engine).values())
        share = round(100 * subagent_cost / total, 1) if total else 0.0
        rows.append(
            {
                "metric": "Subagents",
                "value": f"${subagent_cost:,.2f} ({share}% of {providers[0]} cost)",
            }
        )

    return TableResult(
        title="Usage summary",
        columns=[Column("metric", "Metric"), Column("value", "Value", "str")],
        rows=rows,
        notes=notes,
    )


def by_project(ds: Dataset, provider: str) -> TableResult:
    """Cost/tokens/prompts per project, sorted by cost, with a cumulative %."""
    engine = CostEngine(provider, ds.pricing_path)
    prompt_project = _project_of(ds)
    session_project = _session_project(ds)

    cost: defaultdict[str, float] = defaultdict(float)
    tokens: Counter[str] = Counter()
    for row in ds.tokens:
        project = prompt_project.get(row["prompt_id"]) or session_project.get(row["session_id"], "")
        project = project or "(unknown)"
        cost[project] += engine.cost(row.get("model") or "", row["token_type"], row["token_count"])
        if row["token_type"] != "server_tool_use":
            tokens[project] += row["token_count"]

    prompts: Counter[str] = Counter()
    for row in ds.prompts:
        prompts[(row.get("project") or "(unknown)")] += 1

    total_cost = sum(cost.values())
    total_tokens = sum(tokens.values())
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for project, project_cost in sorted(cost.items(), key=lambda kv: -kv[1]):
        cumulative += project_cost
        out_row: dict[str, Any] = {
            "project": project,
            "prompts": prompts.get(project, 0),
            "tokens": tokens.get(project, 0),
            "token_share_pct": (
                round(100 * tokens.get(project, 0) / total_tokens, 1) if total_tokens else 0.0
            ),
            "cost_usd": round(project_cost, 4),
            "share_pct": round(100 * project_cost / total_cost, 1) if total_cost else 0.0,
            "cumulative_pct": round(100 * cumulative / total_cost, 1) if total_cost else 0.0,
        }
        rows.append(out_row)

    columns = [
        Column("project", "Project"),
        Column("prompts", "Prompts", "int"),
        Column("tokens", "Tokens", "int"),
        Column("token_share_pct", "Token %", "pct"),
        Column("cost_usd", f"Cost ({provider})", "money"),
        Column("share_pct", "Cost %", "pct"),
        Column("cumulative_pct", "Cumulative", "pct"),
    ]
    notes = [_source_note(ds)]
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult("Cost by project", columns, rows, notes)


def by_model(ds: Dataset, provider: str, *, compact: bool = False) -> TableResult:
    """Token and cost split per model.

    Cache writes stay split by TTL (1.3: the 1h writes are billed 2x and can
    dominate 10:1 on real data -- merging them hides the driver) and the
    subagent share of each model's cost gets its own column (1.2).

    ``compact`` (2.8) drops the input/output columns and abbreviates token
    counts (``1.01G``) so the table fits an 80-column terminal instead of
    folding every cell -- the cost-driver columns (cache read/write, subagents,
    cost) are kept.
    """
    engine = CostEngine(provider, ds.pricing_path)
    real = _real_prompt_ids(ds)

    cost: defaultdict[str, float] = defaultdict(float)
    subagent_cost: defaultdict[str, float] = defaultdict(float)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_sets: dict[str, set[str]] = defaultdict(set)
    for row in ds.tokens:
        model = row.get("model") or "(unknown)"
        row_cost = engine.cost(row.get("model") or "", row["token_type"], row["token_count"])
        cost[model] += row_cost
        if row.get("is_sidechain"):
            subagent_cost[model] += row_cost
        counts[model][row["token_type"]] += row["token_count"]
        if row["prompt_id"] in real:
            prompt_sets[model].add(row["prompt_id"])

    total_cost = sum(cost.values())
    rows: list[dict[str, Any]] = []
    for model, model_cost in sorted(cost.items(), key=lambda kv: -kv[1]):
        c = counts[model]
        rows.append(
            {
                "model": model,
                "prompts": len(prompt_sets.get(model, set())),
                "input": c.get("input", 0),
                "output": c.get("output", 0),
                "cache_read": c.get("cache_read", 0),
                "cache_write_5m": c.get("cache_write_5m", 0),
                "cache_write_1h": c.get("cache_write_1h", 0),
                "subagent_cost_usd": round(subagent_cost.get(model, 0.0), 4),
                "cost_usd": round(model_cost, 4),
                "share_pct": round(100 * model_cost / total_cost, 1) if total_cost else 0.0,
            }
        )

    notes = [
        _source_note(ds),
        "Cache writes are split by TTL: 5m is billed 1.25x input, 1h is billed 2x.",
    ]
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    if compact:
        # Short labels + abbreviated tokens + only the cost-driver columns, so
        # the long model names keep enough room to stay readable at 80 cols.
        columns = [
            Column("model", "Model"),
            Column("prompts", "Prm", "int"),
            Column("cache_read", "Reads", "tokens"),
            Column("cache_write_1h", "Wr1h", "tokens"),
            Column("subagent_cost_usd", "Subag$", "money"),
            Column("cost_usd", "Cost", "money"),
            Column("share_pct", "Share", "pct"),
        ]
    else:
        columns = [
            Column("model", "Model"),
            Column("prompts", "Prompts", "int"),
            Column("input", "Input", "int"),
            Column("output", "Output", "int"),
            Column("cache_read", "Cache read", "int"),
            Column("cache_write_5m", "Cache write 5m", "int"),
            Column("cache_write_1h", "Cache write 1h", "int"),
            Column("subagent_cost_usd", "Subagents", "money"),
            Column("cost_usd", f"Cost ({provider})", "money"),
            Column("share_pct", "Share", "pct"),
        ]
    return TableResult("Cost by model", columns, rows, notes)


def by_token_type(ds: Dataset, provider: str) -> TableResult:
    """Token volume and cost split per token type: the cost-driver view.

    On real Claude Code usage most of the bill is *context rent* (cache
    reads + cache writes -- money spent re-sending context), not generation.
    That share is computed here and surfaced in the notes; it is the single
    number that tells where to optimize. ``server_tool_use`` rows are shown
    (requests) but never priced.
    """
    engine = CostEngine(provider, ds.pricing_path)
    counts: Counter[str] = Counter()
    cost: defaultdict[str, float] = defaultdict(float)
    for row in ds.tokens:
        token_type = row["token_type"]
        counts[token_type] += row["token_count"]
        cost[token_type] += engine.cost(row.get("model") or "", token_type, row["token_count"])

    total_cost = sum(cost.values())
    total_tokens = _token_total(counts)

    def _share(value: float) -> float:
        return round(100 * value / total_cost, 1) if total_cost else 0.0

    rows: list[dict[str, Any]] = []
    for token_type in TOKEN_TYPES:
        if not counts.get(token_type):
            continue
        label = TOKEN_TYPE_LABELS[token_type]
        # server_tool_use is counted in requests, not tokens: a token share
        # would be meaningless, so leave it blank.
        token_share = (
            None
            if token_type == "server_tool_use"
            else (round(100 * counts[token_type] / total_tokens, 1) if total_tokens else 0.0)
        )
        if token_type == "server_tool_use":
            label += " (requests, billed separately)"
        rows.append(
            {
                "token_type": label,
                "tokens": counts[token_type],
                "token_share_pct": token_share,
                "cost_usd": round(cost.get(token_type, 0.0), 4),
                "cost_share_pct": _share(cost.get(token_type, 0.0)),
            }
        )
    rows.sort(key=lambda r: -r["cost_usd"])
    rows.append(
        {
            "token_type": "TOTAL",
            "tokens": total_tokens,
            "token_share_pct": 100.0 if total_tokens else 0.0,
            "cost_usd": round(total_cost, 4),
            "cost_share_pct": 100.0 if total_cost else 0.0,
        }
    )

    notes = [_source_note(ds)]
    if total_cost:
        rent = sum(cost.get(t, 0.0) for t in ("cache_read", "cache_write_5m", "cache_write_1h"))
        notes.append(
            f"Context rent (cache reads + writes): {_share(rent)}% of the bill; "
            f"generation (output): {_share(cost.get('output', 0.0))}%; "
            f"fresh input: {_share(cost.get('input', 0.0))}%."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Cost by token type ({provider})",
        [
            Column("token_type", "Token type"),
            Column("tokens", "Tokens", "int"),
            Column("token_share_pct", "Token %", "pct"),
            Column("cost_usd", "Cost", "money"),
            Column("cost_share_pct", "Cost %", "pct"),
        ],
        rows,
        notes,
    )


def by_category(ds: Dataset, provider: str) -> TableResult:
    """Cost/prompt split per LLM-assigned category (needs ``categorize``)."""
    engine = CostEngine(provider, ds.pricing_path)
    prompt_costs = _prompt_costs(ds, engine)
    real = _real_prompt_ids(ds)

    cost: defaultdict[str, float] = defaultdict(float)
    per_prompt_costs: dict[str, list[float]] = defaultdict(list)
    prompts: Counter[str] = Counter()
    complexities: dict[str, list[int]] = defaultdict(list)
    for row in ds.prompts:
        pid = row["prompt_id"]
        info = ds.categories.get(pid)
        category = (info or {}).get("category") or "(uncategorized)"
        prompt_cost = prompt_costs.get(pid, 0.0)
        cost[category] += prompt_cost
        per_prompt_costs[category].append(prompt_cost)
        prompts[category] += 1
        complexity = (info or {}).get("complexity", "")
        if complexity.isdigit():
            complexities[category].append(int(complexity))

    overhead = sum(c for pid, c in prompt_costs.items() if pid not in real)
    total_prompts = sum(prompts.values())
    total_cost = sum(cost.values())
    rows: list[dict[str, Any]] = []
    for category, category_cost in sorted(cost.items(), key=lambda kv: -kv[1]):
        levels = complexities.get(category, [])
        costs_here = per_prompt_costs.get(category, [])
        rows.append(
            {
                "category": category,
                "prompts": prompts[category],
                "share_pct": (
                    round(100 * prompts[category] / total_prompts, 1) if total_prompts else 0.0
                ),
                "avg_complexity": round(sum(levels) / len(levels), 1) if levels else None,
                "med_cost_per_prompt_usd": (
                    round(statistics.median(costs_here), 4) if costs_here else None
                ),
                "cost_usd": round(category_cost, 4),
                "cost_share_pct": (
                    round(100 * category_cost / total_cost, 1) if total_cost else 0.0
                ),
            }
        )

    notes = [_source_note(ds)]
    if not ds.categories:
        notes.append(
            "No categorization found -- run `prompt-analytics categorize` to fill this view."
        )
    if overhead:
        notes.append(
            f"Session overhead (continuations, compactions): ${overhead:,.2f} "
            "not attributable to a prompt, excluded above."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        "Cost by category",
        [
            Column("category", "Category"),
            Column("prompts", "Prompts", "int"),
            Column("share_pct", "Prompt %", "pct"),
            Column("avg_complexity", "Avg complexity", "num"),
            Column("med_cost_per_prompt_usd", "$/prompt (med)", "money"),
            Column("cost_usd", f"Cost ({provider})", "money"),
            Column("cost_share_pct", "Cost %", "pct"),
        ],
        rows,
        notes,
    )


def input_cost(ds: Dataset, provider: str) -> float:
    """Cost of *fresh input* tokens only (token type ``input``), priced on ``provider``.

    The Composition "Input cost" KPI must price what you actually *sent fresh*
    each turn, not the whole prompt. :func:`by_category` sums **every** token type
    (input + output + all cache), i.e. ~the entire bill -- so reusing its
    ``cost_usd`` for an "Input cost" headline overstates it by orders of
    magnitude. This isolates the ``input`` rows, so the figure reconciles to the
    Overview's fresh-input number (same token type, same :class:`CostEngine`).
    Metrics only.
    """
    engine = CostEngine(provider, ds.pricing_path)
    return round(
        sum(
            engine.cost(row.get("model") or "", "input", row["token_count"])
            for row in ds.tokens
            if row["token_type"] == "input"
        ),
        4,
    )


def _output_cost_by_prompt(ds: Dataset, engine: CostEngine) -> dict[str, float]:
    """``prompt_id -> USD`` of the generated **output** tokens only (3.x)."""
    costs: dict[str, float] = defaultdict(float)
    for row in ds.tokens:
        if row["token_type"] == "output":
            costs[row["prompt_id"]] += engine.cost(
                row.get("model") or "", "output", row["token_count"]
            )
    return costs


@dataclass(frozen=True)
class LanguageComposition:
    """One language's slice of what the assistant produced (Axe C).

    ``test_added`` is the share of ``lines_added`` that landed in tests;
    ``code_cost`` is the output spend attributed to this language (the code
    half of the prose/code split, distributed across the languages a prompt
    edited by line churn).
    """

    language: str
    files: int
    lines_added: int
    lines_deleted: int
    test_added: int
    code_cost: float


@dataclass(frozen=True)
class OutputComposition:
    """Structured Axe-C output-composition metrics (shared by CLI + dashboard).

    ``languages`` is sorted by lines produced (added), descending. The cost
    split prorates each prompt's real ``output`` cost by its local-tokenizer
    prose/code weight (the honest estimate the plan settled on); the code half
    is then attributed across the languages a prompt edited, by line churn.
    Code spend with no file edit (Bash / Read / Grep only) lands in
    ``tooling_cost``, so the per-language costs plus ``tooling_cost`` reconcile
    to ``code_cost``. Metrics only -- no source code is read here.
    """

    provider: str
    languages: list[LanguageComposition]
    total_files: int
    total_added: int
    total_deleted: int
    total_test: int
    prose_tokens: int
    code_tokens: int
    prose_cost: float
    code_cost: float
    tooling_cost: float

    @property
    def has_data(self) -> bool:
        """True when there is any line-diff or prose/code-token metric to show."""
        return bool(self.languages) or bool(self.prose_tokens or self.code_tokens)


def output_composition(ds: Dataset, provider: str) -> OutputComposition:
    """Compute the Axe-C output-composition metrics (see :class:`OutputComposition`).

    Aggregates the per-prompt file-edit rows by language (lines +/-, files,
    test share) and prorates the generated output cost into a prose half and a
    code half, the latter attributed back to languages by line churn. The pure
    numbers feed both :func:`by_output` (the CLI table + notes) and the
    dashboard's Composition view, so the two never drift.
    """
    added: Counter[str] = Counter()
    deleted: Counter[str] = Counter()
    paths_by_lang: dict[str, set[str]] = defaultdict(set)
    test_added: Counter[str] = Counter()
    # Per-prompt language churn (added + deleted), the weight used to attribute
    # each prompt's code cost across the languages it touched.
    prompt_churn: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ds.output_files:
        language = row.get("language") or "(unknown)"
        la = int(row.get("lines_added") or 0)
        ld = int(row.get("lines_deleted") or 0)
        added[language] += la
        deleted[language] += ld
        # One row per file (path); count distinct files so a file edited across
        # several prompts is still one file touched.
        paths_by_lang[language].add(row.get("path") or "")
        if (row.get("kind") or "") == "test":
            test_added[language] += la
        prompt_churn[row["prompt_id"]][language] += la + ld
    files: Counter[str] = Counter({lang: len(paths) for lang, paths in paths_by_lang.items()})

    engine = CostEngine(provider, ds.pricing_path)
    out_cost = _output_cost_by_prompt(ds, engine)

    prose_tokens = code_tokens = 0
    prose_cost = code_cost = tooling_cost = 0.0
    lang_cost: dict[str, float] = defaultdict(float)
    for row in ds.output_tokens:
        prose = int(row.get("output_prose_tokens") or 0)
        code = int(row.get("output_code_tokens") or 0)
        prose_tokens += prose
        code_tokens += code
        weight = prose + code
        pid = row["prompt_id"]
        pid_cost = out_cost.get(pid, 0.0)
        prose_share = pid_cost * prose / weight if weight else pid_cost
        code_share = pid_cost - prose_share
        prose_cost += prose_share
        code_cost += code_share
        churn = prompt_churn.get(pid)
        total_churn = sum(churn.values()) if churn else 0
        if churn and total_churn:
            for language, c in churn.items():
                lang_cost[language] += code_share * c / total_churn
        else:
            # Code tokens with no file edit (Bash / Read / Grep only): no
            # language to attribute to, so they form an honest tooling bucket.
            tooling_cost += code_share

    languages = [
        LanguageComposition(
            language=language,
            files=files.get(language, 0),
            lines_added=lines_added,
            lines_deleted=deleted.get(language, 0),
            test_added=test_added.get(language, 0),
            code_cost=round(lang_cost.get(language, 0.0), 6),
        )
        for language, lines_added in sorted(added.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return OutputComposition(
        provider=provider,
        languages=languages,
        total_files=sum(files.values()),
        total_added=sum(added.values()),
        total_deleted=sum(deleted.values()),
        total_test=sum(test_added.values()),
        prose_tokens=prose_tokens,
        code_tokens=code_tokens,
        prose_cost=round(prose_cost, 6),
        code_cost=round(code_cost, 6),
        tooling_cost=round(tooling_cost, 6),
    )


def by_output(ds: Dataset, provider: str) -> TableResult:
    """Output composition (Axe C): what the assistant actually produced.

    The main table is the **language mix** (one row per language, sorted by
    lines produced) with, per language, the files touched, the exact +/- line
    diff, and the share of those added lines that landed in **tests** vs code.
    Two headline notes carry the cross-language story: the overall test ratio of
    the produced lines, and the **prose vs code** split of the generated output
    tokens priced on ``provider`` (each prompt's output cost prorated by its
    local-tokenizer prose/code weight). Metrics only -- no source code is ever
    read here, just the integer rows ``extract`` derived. The pure computation
    lives in :func:`output_composition` (shared with the dashboard).
    """
    comp = output_composition(ds, provider)
    total_added = comp.total_added
    total_test = comp.total_test

    rows: list[dict[str, Any]] = []
    for lang in comp.languages:
        rows.append(
            {
                "language": lang.language,
                "files": lang.files,
                "lines_added": lang.lines_added,
                "lines_deleted": lang.lines_deleted,
                "test_pct": round(100 * lang.test_added / lang.lines_added, 1)
                if lang.lines_added
                else 0.0,
                "share_pct": round(100 * lang.lines_added / total_added, 1) if total_added else 0.0,
            }
        )
    if rows:
        rows.append(
            {
                "language": "TOTAL",
                "files": comp.total_files,
                "lines_added": total_added,
                "lines_deleted": comp.total_deleted,
                "test_pct": round(100 * total_test / total_added, 1) if total_added else 0.0,
                "share_pct": 100.0 if total_added else 0.0,
            }
        )

    notes = [_source_note(ds)]
    if not ds.output_files and not ds.output_tokens:
        notes.append(
            "No output-composition data -- re-run `prompt-analytics extract` "
            "(these metrics ship with the latest extractor)."
        )

    engine = CostEngine(provider, ds.pricing_path)
    if total_added:
        notes.append(
            f"Code vs tests: {round(100 * total_test / total_added, 1)}% of the "
            f"{total_added:,} added lines are tests "
            f"({total_added - total_test:,} code, {total_test:,} test)."
        )
    if comp.prose_tokens or comp.code_tokens:
        gen_cost = comp.prose_cost + comp.code_cost
        code_cost_share = round(100 * comp.code_cost / gen_cost, 1) if gen_cost else 0.0
        notes.append(
            f"Generated output: {comp.prose_tokens:,} prose tokens (${comp.prose_cost:,.2f}) vs "
            f"{comp.code_tokens:,} code/tool tokens (${comp.code_cost:,.2f}) -- "
            f"{code_cost_share}% of generation cost is code."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)

    return TableResult(
        f"Output composition ({provider})",
        [
            Column("language", "Language"),
            Column("files", "Files", "int"),
            Column("lines_added", "Lines +", "int"),
            Column("lines_deleted", "Lines −", "int"),
            Column("test_pct", "Test %", "pct"),
            Column("share_pct", "Lines %", "pct"),
        ],
        rows,
        notes,
    )


# ---------------------------------------------------------------------------
# Axe D: context cost over time (D2) -- the real cost of a context element.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextElementCost:
    """One context element's cost split into loading vs rent (Axe D).

    ``rent_cost`` is the priced ``cache_read`` the element paid for staying in
    context turn after turn (size x turns of presence); ``load_cost`` the one-off
    ``cache_write`` of caching it. ``language`` is meaningful for ``file_read``
    (``-`` otherwise).

    The raw token counts behind those prices ride alongside (``rent_read_tokens``
    = the ``cache_read`` tokens; ``load_write_5m_tokens``/``load_write_1h_tokens``
    = the ``cache_write`` tokens by TTL) so a view can plot the *size* of what
    lingers in tokens, not only its provider-priced dollars.
    """

    source: str
    language: str
    load_cost: float
    rent_cost: float
    rent_read_tokens: int
    load_write_5m_tokens: int
    load_write_1h_tokens: int

    @property
    def total_cost(self) -> float:
        return self.load_cost + self.rent_cost

    @property
    def load_tokens(self) -> int:
        """One-off cache-write tokens of caching this element (5m + 1h)."""
        return self.load_write_5m_tokens + self.load_write_1h_tokens

    @property
    def total_tokens(self) -> int:
        """Every token this element drove: rent (re-reads) + loading (writes)."""
        return self.rent_read_tokens + self.load_tokens


@dataclass(frozen=True)
class ContextCost:
    """Structured Axe-D context cost-over-time metrics (shared by CLI + dashboard).

    ``elements`` is sorted by total cost (descending), the real cache spend each
    context source/language drove. The headline split is loading (one-off cache
    writes) vs rent (cache reads paid every turn the element stays in context).
    ``unattributed_cost`` is cache that landed on a turn with no measured element
    (the post-compaction summary, pre-context turns): it keeps the total honest
    so ``load + rent`` reconciles to the billed main-chain cache cost exactly.
    """

    provider: str
    elements: list[ContextElementCost]
    load_cost: float
    rent_cost: float
    unattributed_cost: float
    rent_read_tokens: int
    load_write_5m_tokens: int
    load_write_1h_tokens: int

    @property
    def total_cost(self) -> float:
        return self.load_cost + self.rent_cost

    @property
    def attributed_cost(self) -> float:
        return self.total_cost - self.unattributed_cost

    @property
    def load_tokens(self) -> int:
        """Total one-off cache-write tokens across the attributed elements."""
        return self.load_write_5m_tokens + self.load_write_1h_tokens

    @property
    def total_tokens(self) -> int:
        """Total context tokens (rent re-reads + loading writes), attributed only."""
        return self.rent_read_tokens + self.load_tokens

    @property
    def has_data(self) -> bool:
        return bool(self.elements)


def context_cost(ds: Dataset, provider: str) -> ContextCost:
    """Compute the Axe-D context cost-over-time metrics (see :class:`ContextCost`).

    Prices the per-element cache tokens ``extract`` attributed (rent = the real
    ``cache_read`` spread by size x turns of presence; load = the one-off
    ``cache_write``) on ``provider``, aggregating per ``(source, language)``
    across models. The pure numbers feed both :func:`by_context` (the CLI table)
    and the dashboard, so the two never drift. Metrics only -- no content is read.
    """
    engine = CostEngine(provider, ds.pricing_path)
    load: dict[tuple[str, str], float] = defaultdict(float)
    rent: dict[tuple[str, str], float] = defaultdict(float)
    # Raw token counts behind the prices, same (source, language) key, so a view
    # can plot the size of context in tokens (D2 / Prompt 3 token charts).
    rent_tok: dict[tuple[str, str], int] = defaultdict(int)
    load5m_tok: dict[tuple[str, str], int] = defaultdict(int)
    load1h_tok: dict[tuple[str, str], int] = defaultdict(int)
    for row in ds.context_cost:
        model = row.get("model") or ""
        key = (row.get("source") or "", row.get("language") or NO_LANGUAGE)
        read = int(row.get("rent_read_tokens") or 0)
        write5m = int(row.get("load_write_5m_tokens") or 0)
        write1h = int(row.get("load_write_1h_tokens") or 0)
        rent_tok[key] += read
        load5m_tok[key] += write5m
        load1h_tok[key] += write1h
        rent[key] += engine.cost(model, "cache_read", read)
        load[key] += engine.cost(model, "cache_write_5m", write5m) + engine.cost(
            model, "cache_write_1h", write1h
        )

    keys = set(load) | set(rent)
    elements = [
        ContextElementCost(
            source=source,
            language=language,
            load_cost=round(load.get((source, language), 0.0), 6),
            rent_cost=round(rent.get((source, language), 0.0), 6),
            rent_read_tokens=rent_tok.get((source, language), 0),
            load_write_5m_tokens=load5m_tok.get((source, language), 0),
            load_write_1h_tokens=load1h_tok.get((source, language), 0),
        )
        for source, language in keys
        if source != UNATTRIBUTED_SOURCE
    ]
    elements.sort(key=lambda e: (-e.total_cost, e.source, e.language))
    unattributed = sum(
        load.get(k, 0.0) + rent.get(k, 0.0) for k in keys if k[0] == UNATTRIBUTED_SOURCE
    )

    def _attr_token_sum(per_key: dict[tuple[str, str], int]) -> int:
        return sum(v for k, v in per_key.items() if k[0] != UNATTRIBUTED_SOURCE)

    return ContextCost(
        provider=provider,
        elements=elements,
        load_cost=round(sum(load.values()), 6),
        rent_cost=round(sum(rent.values()), 6),
        unattributed_cost=round(unattributed, 6),
        rent_read_tokens=_attr_token_sum(rent_tok),
        load_write_5m_tokens=_attr_token_sum(load5m_tok),
        load_write_1h_tokens=_attr_token_sum(load1h_tok),
    )


# Source -> human label for the context cost table (the four-bucket taxonomy).
_CONTEXT_SOURCE_LABELS = {
    "conversation": "Conversation",
    "file_read": "Files read",
    "tool_output": "Tool output",
    "config": "Config / setup",
    UNATTRIBUTED_SOURCE: "(unattributed)",
}


def _main_chain_cache_cost(ds: Dataset, engine: CostEngine) -> float:
    """The billed cache cost (read + writes) of every non-sidechain request.

    The ground truth the attributed context cost reconciles to: the bill for the
    main conversation's context (subagent requests carry their own context and
    are excluded from the snapshot, hence from here too)."""
    total = 0.0
    for row in ds.requests:
        if row.get("is_sidechain"):
            continue
        model = row.get("model") or ""
        total += engine.cost(model, "cache_read", int(row.get("cache_read_tokens") or 0))
        total += _request_write_cost(engine, row)
    return total


def by_context(ds: Dataset, provider: str) -> TableResult:
    """Context cost over time (Axe D): the real cost of what fills the context.

    Each context source/language is one row, split into the one-off **loading**
    cost (cache writes) and the **rent** it pays every turn it stays in context
    (cache reads = size x turns of presence) -- the differentiator: "this file
    cost $X to load and $Y of rent". The attributed totals reconcile to the
    billed main-chain cache cost to the dollar; cache that cannot be tied to a
    measured element (chiefly post-compaction summaries -- the ``parentUuid`` ~=
    API-context caveat) is shown honestly as ``(unattributed)``. Metrics only.
    The pure computation lives in :func:`context_cost` (shared with the dashboard).
    """
    comp = context_cost(ds, provider)
    engine = CostEngine(provider, ds.pricing_path)

    rows: list[dict[str, Any]] = []
    total = comp.total_cost
    for element in comp.elements:
        rows.append(
            {
                "source": _CONTEXT_SOURCE_LABELS.get(element.source, element.source),
                "language": element.language,
                "load_usd": round(element.load_cost, 4),
                "rent_usd": round(element.rent_cost, 4),
                "total_usd": round(element.total_cost, 4),
                "share_pct": round(100 * element.total_cost / total, 1) if total else 0.0,
            }
        )
    if comp.unattributed_cost:
        rows.append(
            {
                "source": _CONTEXT_SOURCE_LABELS[UNATTRIBUTED_SOURCE],
                "language": NO_LANGUAGE,
                "load_usd": None,
                "rent_usd": None,
                "total_usd": round(comp.unattributed_cost, 4),
                "share_pct": round(100 * comp.unattributed_cost / total, 1) if total else 0.0,
            }
        )
    if rows:
        rows.append(
            {
                "source": "TOTAL",
                "language": "",
                "load_usd": round(comp.load_cost, 4),
                "rent_usd": round(comp.rent_cost, 4),
                "total_usd": round(total, 4),
                "share_pct": 100.0 if total else 0.0,
            }
        )

    notes = [_source_note(ds)]
    if not ds.context_cost:
        notes.append(
            "No context-cost data -- re-run `prompt-analytics extract` "
            "(these metrics ship with the latest extractor)."
        )
    if comp.has_data and total:
        rent_share = round(100 * comp.rent_cost / total, 1)
        notes.append(
            f"Loading vs rent: ${comp.load_cost:,.2f} to cache the context once, "
            f"${comp.rent_cost:,.2f} of rent re-reading it every turn it stays "
            f"({rent_share}% of the cache bill is rent -- the cost of context that "
            "lingers; that is what /compact and a leaner CLAUDE.md cut)."
        )
        top = comp.elements[0]
        notes.append(
            f"Top context cost: {_CONTEXT_SOURCE_LABELS.get(top.source, top.source)}"
            + (f" ({top.language})" if top.language != NO_LANGUAGE else "")
            + f" at ${top.total_cost:,.2f} "
            f"(${top.load_cost:,.2f} load + ${top.rent_cost:,.2f} rent)."
        )
        bill = _main_chain_cache_cost(ds, engine)
        unattr_share = round(100 * comp.unattributed_cost / total, 1) if total else 0.0
        notes.append(
            f"Reconciliation: attributed ${comp.attributed_cost:,.2f} + "
            f"${comp.unattributed_cost:,.2f} unattributed = ${total:,.2f}, the billed "
            f"main-chain cache cost (${bill:,.2f}). The {unattr_share}% unattributed is "
            "cache on turns with no measured element (post-compaction summaries / "
            "pre-context turns): the parentUuid chain approximates the API context."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)

    return TableResult(
        f"Context cost over time ({provider})",
        [
            Column("source", "Source"),
            Column("language", "Language"),
            Column("load_usd", "Load $", "money"),
            Column("rent_usd", "Rent $", "money"),
            Column("total_usd", "Total $", "money"),
            Column("share_pct", "Share", "pct"),
        ],
        rows,
        notes,
    )


# ---------------------------------------------------------------------------
# Unified per-file view (DASH4): a file's whole footprint, Axe C joined to D.
# ---------------------------------------------------------------------------


def file_footprint(ds: Dataset, provider: str) -> TableResult:
    """One row per file crossing Axe C (edits) and Axe D (reads + context cost).

    The Explorer-style scorecard: a file's total cost of ownership in a single
    line -- how often it was **edited** and its exact +/- line diff (Axe C),
    how often it was **read** into context and what that context cost (the one-off
    **load** plus the **rent** it paid every turn it lingered, Axe D), keyed on
    the project-relative path so a file edited *and* kept in context shows both
    halves. Rows are sorted by context cost (the actionable "what to cut") then
    line churn. Metrics only -- relative paths, never a byte of content.
    """
    engine = CostEngine(provider, ds.pricing_path)

    edits: Counter[str] = Counter()
    added: Counter[str] = Counter()
    deleted: Counter[str] = Counter()
    language: dict[str, str] = {}
    kind: dict[str, str] = {}

    def _note_lang(path: str, value: str | None) -> None:
        if value and value != NO_LANGUAGE and path not in language:
            language[path] = value

    for row in ds.output_files:
        path = row.get("path") or ""
        if not path or path == NO_PATH:
            continue
        edits[path] += int(row.get("edits") or 0)
        added[path] += int(row.get("lines_added") or 0)
        deleted[path] += int(row.get("lines_deleted") or 0)
        _note_lang(path, row.get("language"))
        if path not in kind:
            kind[path] = row.get("kind") or detect_kind(path)

    reads: Counter[str] = Counter()
    for row in ds.context_sources:
        if (row.get("source") or "") != "file_read":
            continue
        path = row.get("path") or ""
        if not path or path == NO_PATH:
            continue
        reads[path] += int(row.get("items") or 0)
        _note_lang(path, row.get("language"))

    load: dict[str, float] = defaultdict(float)
    rent: dict[str, float] = defaultdict(float)
    for row in ds.context_cost:
        if (row.get("source") or "") != "file_read":
            continue
        path = row.get("path") or ""
        if not path or path == NO_PATH:
            continue
        model = row.get("model") or ""
        rent[path] += engine.cost(model, "cache_read", int(row.get("rent_read_tokens") or 0))
        load[path] += engine.cost(
            model, "cache_write_5m", int(row.get("load_write_5m_tokens") or 0)
        ) + engine.cost(model, "cache_write_1h", int(row.get("load_write_1h_tokens") or 0))
        _note_lang(path, row.get("language"))

    all_paths = set(edits) | set(reads) | set(load) | set(rent)
    rows: list[dict[str, Any]] = []
    for path in all_paths:
        context_usd = load.get(path, 0.0) + rent.get(path, 0.0)
        rows.append(
            {
                "path": path,
                "language": language.get(path, NO_LANGUAGE),
                "kind": kind.get(path) or detect_kind(path),
                "edits": edits.get(path, 0),
                "lines_added": added.get(path, 0),
                "lines_deleted": deleted.get(path, 0),
                "reads": reads.get(path, 0),
                "load_usd": round(load.get(path, 0.0), 4),
                "rent_usd": round(rent.get(path, 0.0), 4),
                "context_usd": round(context_usd, 4),
            }
        )
    rows.sort(
        key=lambda r: (-r["context_usd"], -(r["lines_added"] + r["lines_deleted"]), r["path"])
    )

    notes = [_source_note(ds)]
    if not ds.output_files and not ds.context_cost:
        notes.append(
            "No per-file data -- re-run `prompt-analytics extract` "
            "(the file identity ships with the latest extractor)."
        )
    elif rows:
        edited = sum(1 for r in rows if r["edits"])
        read_only = sum(1 for r in rows if not r["edits"] and r["reads"])
        notes.append(
            f"{len(rows):,} files: {edited:,} edited, {read_only:,} read but never edited "
            "(pure context cost -- the first candidates to keep out of context)."
        )

    return TableResult(
        f"Per-file footprint ({provider})",
        [
            Column("path", "File"),
            Column("language", "Language"),
            Column("kind", "Kind"),
            Column("edits", "Edits", "int"),
            Column("lines_added", "Lines +", "int"),
            Column("lines_deleted", "Lines −", "int"),
            Column("reads", "Reads", "int"),
            Column("load_usd", "Load $", "money"),
            Column("rent_usd", "Rent $", "money"),
            Column("context_usd", "Context $", "money"),
        ],
        rows,
        notes,
    )


# ---------------------------------------------------------------------------
# Axe B2: cost by task -- the task, not the prompt, is the unit of work.
# ---------------------------------------------------------------------------

# Token types that are context (cache) rather than fresh generation: the share
# of a task's cost spent re-reading / caching context, the B2 "part contexte".
_CONTEXT_TOKEN_TYPES = frozenset({"cache_read", "cache_write_5m", "cache_write_1h"})


def _parse_iso(value: str) -> datetime | None:
    """An aware-or-naive datetime from an ISO-8601 string, or None (empty-safe)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_duration(seconds: float) -> str:
    """A compact human span (e.g. ``2h 15m``); ``<1m`` under a minute."""
    total = int(seconds)
    if total <= 0:
        return "<1m"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def by_task(ds: Dataset, provider: str, *, top: int = 20) -> TableResult:
    """Cost by task (Axe B2): the task is the unit of work, not the prompt.

    Every real prompt belongs to exactly one task (``task_prompts``), so a task's
    cost is just its prompts' token rows priced at read time -- it reconciles to
    the bill by construction (continuation/compaction overhead, tied to no prompt,
    is excluded and surfaced in the notes). Each row carries the task's total cost
    and the **context share** of it (the cache reads/writes its prompts paid -- the
    B2 "this task cost $X, of which Y% is context rent"), its prompt count, its
    span (a task lives within one session -- ``task_id`` is session-scoped -- so
    the span is its first->last prompt duration), its ``origin`` (``todo`` spine vs
    ``inferred``, so readers can weigh it) and its **dominant category** (the most
    frequent category among its prompts, when ``categorize`` has run). Sorted by
    cost, top ``top`` (0 = all). Metrics only -- the task name is a Claude-authored
    todo label or a blanked-under-``--no-text`` snippet, never source content.
    """
    engine = CostEngine(provider, ds.pricing_path)
    task_of = {row["prompt_id"]: row["task_id"] for row in ds.task_prompts}

    total_cost: defaultdict[str, float] = defaultdict(float)
    context_cost_by_task: defaultdict[str, float] = defaultdict(float)
    overhead = 0.0
    for row in ds.tokens:
        c = engine.cost(row.get("model") or "", row["token_type"], row["token_count"])
        tid = task_of.get(row["prompt_id"])
        if tid is None:
            overhead += c
            continue
        total_cost[tid] += c
        if row["token_type"] in _CONTEXT_TOKEN_TYPES:
            context_cost_by_task[tid] += c

    dominant: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ds.task_prompts:
        info = ds.categories.get(row["prompt_id"])
        category = (info or {}).get("category") or ""
        if category:
            dominant[row["task_id"]][category] += 1

    grand_total = sum(total_cost.values())
    context_total = sum(context_cost_by_task.values())
    rows: list[dict[str, Any]] = []
    for task in ds.tasks:
        tid = task["task_id"]
        cost = total_cost.get(tid, 0.0)
        ctx = context_cost_by_task.get(tid, 0.0)
        counter = dominant.get(tid)
        first = _parse_iso(task.get("first_timestamp", ""))
        last = _parse_iso(task.get("last_timestamp", ""))
        span = _format_duration((last - first).total_seconds()) if first and last else ""
        rows.append(
            {
                "task": (task.get("name") or tid)[:60],
                "origin": task.get("origin", ""),
                "prompts": int(task.get("prompts") or 0),
                "duration": span,
                "category": counter.most_common(1)[0][0] if counter else "(uncategorized)",
                "context_pct": round(100 * ctx / cost, 1) if cost else 0.0,
                "cost_usd": round(cost, 4),
                "cost_share_pct": round(100 * cost / grand_total, 1) if grand_total else 0.0,
            }
        )
    rows.sort(key=lambda r: (-r["cost_usd"], r["task"]))
    if top:
        rows = rows[:top]

    notes = [_source_note(ds)]
    if not ds.tasks:
        notes.append(
            "No task data -- re-run `prompt-analytics extract` "
            "(task attribution ships with the latest extractor)."
        )
    else:
        todo_n = sum(1 for t in ds.tasks if t.get("origin") == "todo")
        sessions_n = len({t.get("session_id") for t in ds.tasks})
        notes.append(
            f"{len(ds.tasks):,} tasks across {sessions_n:,} sessions: {todo_n:,} from the "
            f"TodoWrite spine, {len(ds.tasks) - todo_n:,} inferred (time gap + semantics)."
        )
        if grand_total:
            notes.append(
                f"Context is {round(100 * context_total / grand_total, 1)}% of task cost "
                f"(${context_total:,.2f} of ${grand_total:,.2f}) -- the cache reads/writes the "
                "tasks' prompts paid; per-task share is the Context % column."
            )
        if not ds.categories:
            notes.append(
                "No categorization found -- run `prompt-analytics categorize` to fill the "
                "Top category column."
            )
    if overhead:
        notes.append(
            f"Session overhead (continuations, compactions): ${overhead:,.2f} "
            "not attributable to a task, excluded above."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)

    return TableResult(
        f"Cost by task ({provider})",
        [
            Column("task", "Task"),
            Column("origin", "Origin"),
            Column("prompts", "Prompts", "int"),
            Column("duration", "Span"),
            Column("category", "Top category"),
            Column("context_pct", "Context %", "pct"),
            Column("cost_usd", f"Cost ({provider})", "money"),
            Column("cost_share_pct", "Cost %", "pct"),
        ],
        rows,
        notes,
    )


@dataclass(frozen=True)
class TaskNode:
    """One task as a graph centre (Axe B2): a node sized by cost, hued by category.

    ``cost`` is the task's full spend (input + output + context, reconciled to the
    bill like :func:`by_task`); ``context_pct`` the share of it that is cache
    rent/loading; ``category`` the dominant category among its prompts (the node
    colour). ``name`` is a Claude-authored todo label or a blanked snippet, never
    source content.
    """

    task_id: str
    name: str
    origin: str
    category: str
    prompts: int
    cost: float
    context_pct: float


@dataclass(frozen=True)
class TaskSatellite:
    """One prompt orbiting its task (Axe B2): a small node linked to its centre.

    Coloured by its own category, sized faintly by its own cost, so a task's
    centre shows the category mix of the prompts that served it.
    """

    prompt_id: str
    task_id: str
    category: str
    cost: float


@dataclass(frozen=True)
class TaskGraph:
    """Structured task-graph data (Axe B2), shared shape for the dashboard view.

    ``tasks`` are the top ``top`` task centres by cost; ``satellites`` the prompts
    of exactly those tasks (so the graph stays legible). ``total_tasks`` /
    ``todo_tasks`` describe the whole population behind the shown slice;
    ``grand_total`` / ``context_total`` are the reconciled task spend and its
    context share. The pure numbers mirror :func:`by_task` so the graph and the
    CLI table never drift.
    """

    provider: str
    tasks: list[TaskNode]
    satellites: list[TaskSatellite]
    total_tasks: int
    todo_tasks: int
    grand_total: float
    context_total: float

    @property
    def has_data(self) -> bool:
        """True when there is at least one task centre to draw."""
        return bool(self.tasks)


def task_graph(ds: Dataset, provider: str, *, top: int = 40) -> TaskGraph:
    """Assemble the Axe-B2 task graph: task centres + their prompt satellites.

    Prices every prompt once (reconciling to the bill by construction, like
    :func:`by_task`), aggregates to tasks, keeps the top ``top`` by cost (0 =
    all), then gathers the prompts of *those* tasks as satellites so the force
    layout stays readable on a large corpus. The dashboard turns this into the
    ECharts ``graph``; the data assembly lives here so it stays unit-testable.
    """
    engine = CostEngine(provider, ds.pricing_path)
    task_of = {row["prompt_id"]: row["task_id"] for row in ds.task_prompts}

    prompt_cost: defaultdict[str, float] = defaultdict(float)
    prompt_ctx: defaultdict[str, float] = defaultdict(float)
    for row in ds.tokens:
        c = engine.cost(row.get("model") or "", row["token_type"], row["token_count"])
        pid = row["prompt_id"]
        prompt_cost[pid] += c
        if row["token_type"] in _CONTEXT_TOKEN_TYPES:
            prompt_ctx[pid] += c

    task_cost: defaultdict[str, float] = defaultdict(float)
    task_ctx: defaultdict[str, float] = defaultdict(float)
    for pid, c in prompt_cost.items():
        tid = task_of.get(pid)
        if tid is None:  # continuation/compaction overhead, tied to no task
            continue
        task_cost[tid] += c
        task_ctx[tid] += prompt_ctx.get(pid, 0.0)

    dominant: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ds.task_prompts:
        category = (ds.categories.get(row["prompt_id"]) or {}).get("category") or ""
        if category:
            dominant[row["task_id"]][category] += 1

    grand_total = sum(task_cost.values())
    context_total = sum(task_ctx.values())

    nodes: list[TaskNode] = []
    for task in ds.tasks:
        tid = task["task_id"]
        cost = task_cost.get(tid, 0.0)
        ctx = task_ctx.get(tid, 0.0)
        counter = dominant.get(tid)
        nodes.append(
            TaskNode(
                task_id=tid,
                name=task.get("name") or tid,
                origin=task.get("origin", ""),
                category=counter.most_common(1)[0][0] if counter else "(uncategorized)",
                prompts=int(task.get("prompts") or 0),
                cost=round(cost, 4),
                context_pct=round(100 * ctx / cost, 1) if cost else 0.0,
            )
        )
    nodes.sort(key=lambda n: (-n.cost, n.name))
    if top:
        nodes = nodes[:top]

    kept = {n.task_id for n in nodes}
    satellites = [
        TaskSatellite(
            prompt_id=row["prompt_id"],
            task_id=row["task_id"],
            category=(ds.categories.get(row["prompt_id"]) or {}).get("category")
            or "(uncategorized)",
            cost=round(prompt_cost.get(row["prompt_id"], 0.0), 4),
        )
        for row in ds.task_prompts
        if row["task_id"] in kept
    ]

    return TaskGraph(
        provider=provider,
        tasks=nodes,
        satellites=satellites,
        total_tasks=len(ds.tasks),
        todo_tasks=sum(1 for t in ds.tasks if t.get("origin") == "todo"),
        grand_total=round(grand_total, 4),
        context_total=round(context_total, 4),
    )


def top_prompts(ds: Dataset, provider: str, *, top: int = 10) -> TableResult:
    """The N most expensive prompts, with preview."""
    engine = CostEngine(provider, ds.pricing_path)
    prompt_costs = _prompt_costs(ds, engine)
    token_counts = _prompt_token_counts(ds)

    rows: list[dict[str, Any]] = []
    for row in ds.prompts:
        pid = row["prompt_id"]
        rows.append(
            {
                "date": _parse_day(row.get("timestamp", "")),
                "project": row.get("project") or "",
                "model": row.get("model") or "",
                "tokens": _token_total(token_counts.get(pid, Counter())),
                "cost_usd": round(prompt_costs.get(pid, 0.0), 4),
                "preview": (row.get("prompt_preview") or "")[:80],
            }
        )
    rows.sort(key=lambda r: -r["cost_usd"])
    rows = rows[:top]

    notes = [_source_note(ds)]
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Top {top} prompts by cost ({provider})",
        [
            Column("date", "Date"),
            Column("project", "Project"),
            Column("model", "Model"),
            Column("tokens", "Tokens", "int"),
            Column("cost_usd", "Cost", "money"),
            Column("preview", "Prompt"),
        ],
        rows,
        notes,
    )


def sessions_table(ds: Dataset, provider: str, *, top: int = 20) -> TableResult:
    """Sessions ranked by cost."""
    engine = CostEngine(provider, ds.pricing_path)

    cost: defaultdict[str, float] = defaultdict(float)
    tokens: Counter[str] = Counter()
    for row in ds.tokens:
        cost[row["session_id"]] += engine.cost(
            row.get("model") or "", row["token_type"], row["token_count"]
        )
        if row["token_type"] != "server_tool_use":
            tokens[row["session_id"]] += row["token_count"]
    prompts: Counter[str] = Counter()
    for row in ds.prompts:
        prompts[row["session_id"]] += 1
    meta = {row["session_id"]: row for row in ds.sessions}

    rows: list[dict[str, Any]] = []
    for session_id, session_cost in sorted(cost.items(), key=lambda kv: -kv[1]):
        info = meta.get(session_id, {})
        rows.append(
            {
                "session_id": session_id,
                "start_date": info.get("start_date", ""),
                "project": info.get("project", ""),
                "prompts": prompts.get(session_id, 0),
                "tokens": tokens.get(session_id, 0),
                "cost_usd": round(session_cost, 4),
            }
        )
    if top:
        rows = rows[:top]

    notes = [_source_note(ds)]
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Top sessions by cost ({provider})",
        [
            Column("session_id", "Session"),
            Column("start_date", "Started"),
            Column("project", "Project"),
            Column("prompts", "Prompts", "int"),
            Column("tokens", "Tokens", "int"),
            Column("cost_usd", "Cost", "money"),
        ],
        rows,
        notes,
    )


def session_depth(ds: Dataset, provider: str) -> TableResult:
    """Meta-analysis: marginal prompt cost and cache mix vs position in session.

    For each depth band (the prompt's ``prompt_index`` within its session),
    reports the average cost of one prompt at that depth, the multiplier vs a
    session-opening prompt, the per-turn normalizations and the input-side
    token mix (fresh input vs cache reads vs cache writes). This is THE
    differentiating analysis: it shows what a prompt *actually* costs once a
    session gets deep.

    The per-turn columns answer the question the raw average cannot: deep
    prompts are often small follow-ups (few assistant turns), so their
    average cost *decreases* with depth -- not because context is free.
    ``$/turn`` is the band cost divided by its assistant turns;
    ``cache read/turn`` approximates the context size carried at that depth.
    """
    engine = CostEngine(provider, ds.pricing_path)
    prompt_costs = _prompt_costs(ds, engine)
    token_counts = _prompt_token_counts(ds)

    bands: dict[str, dict[str, Any]] = {
        label: {"prompts": 0, "cost": 0.0, "turns": 0, "tokens": Counter()}
        for _, _, label in DEPTH_BANDS
    }

    for row in ds.prompts:
        index = int(row.get("prompt_index") or 0)
        if index < 1:
            continue
        pid = row["prompt_id"]
        band = bands[_depth_band(index)]
        band["prompts"] += 1
        band["cost"] += prompt_costs.get(pid, 0.0)
        with contextlib.suppress(TypeError, ValueError):
            band["turns"] += int(row.get("assistant_turns") or 0)
        band["tokens"].update(token_counts.get(pid, Counter()))

    base_avg: float | None = None
    rows: list[dict[str, Any]] = []
    for _, _, label in DEPTH_BANDS:
        band = bands[label]
        if not band["prompts"]:
            continue
        avg = band["cost"] / band["prompts"]
        if base_avg is None:
            base_avg = avg
        counts: Counter[str] = band["tokens"]
        input_side = (
            counts.get("input", 0)
            + counts.get("cache_read", 0)
            + counts.get("cache_write_5m", 0)
            + counts.get("cache_write_1h", 0)
        )

        def _share(
            kinds: tuple[str, ...],
            counts: Counter[str] = counts,
            input_side: int = input_side,
        ) -> float:
            if not input_side:
                return 0.0
            return round(100 * sum(counts.get(k, 0) for k in kinds) / input_side, 1)

        turns = int(band["turns"])
        rows.append(
            {
                "depth": label,
                "prompts": band["prompts"],
                "avg_cost_usd": round(avg, 4),
                "vs_depth_1": round(avg / base_avg, 2) if base_avg else None,
                "cost_per_turn_usd": round(band["cost"] / turns, 4) if turns else None,
                "cache_read_per_turn": int(counts.get("cache_read", 0) / turns) if turns else None,
                "cache_read_pct": _share(("cache_read",)),
                "cache_write_5m_pct": _share(("cache_write_5m",)),
                "cache_write_1h_pct": _share(("cache_write_1h",)),
                "fresh_input_pct": _share(("input",)),
            }
        )

    notes = [
        _source_note(ds),
        "Depth = the prompt's position within its session (prompt_index). "
        "Shares are over input-side tokens (input + cache read + cache write); "
        "cache writes are split by TTL (5m billed 1.25x input, 1h billed 2x).",
        "$/turn = band cost / assistant turns (deep prompts are often small "
        "follow-ups, so the raw average understates them); cache read/turn "
        "approximates the context size carried at that depth.",
    ]
    if len(rows) > 1 and rows[0]["avg_cost_usd"]:
        deepest = rows[-1]
        notes.append(
            f"A prompt at depth {deepest['depth']} costs x{deepest['vs_depth_1']} "
            "a session-opening prompt."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Marginal prompt cost by session depth ({provider})",
        [
            Column("depth", "Depth"),
            Column("prompts", "Prompts", "int"),
            Column("avg_cost_usd", "Avg cost/prompt", "money"),
            Column("vs_depth_1", "vs depth 1", "x"),
            Column("cost_per_turn_usd", "$/turn", "money"),
            Column("cache_read_per_turn", "Cache read/turn", "int"),
            Column("cache_read_pct", "Cache read", "pct"),
            Column("cache_write_5m_pct", "Cache write 5m", "pct"),
            Column("cache_write_1h_pct", "Cache write 1h", "pct"),
            Column("fresh_input_pct", "Fresh input", "pct"),
        ],
        rows,
        notes,
    )


# ---------------------------------------------------------------------------
# Request-grain analyses (phase 2): accumulated context (2.1), TTL expiry
# losses (2.2), compaction (2.3), fixed session overhead (2.4). They all work
# on ``ds.requests`` (V10) and exclude sidechains: subagents run their own
# context window in parallel, so they belong to neither the main chain's
# growth nor its pauses.
# ---------------------------------------------------------------------------

_NO_REQUESTS_NOTE = (
    "No request-grain data: this analysis needs requests.csv "
    "(re-run `prompt-analytics extract` to produce the v2 schema)."
)

# requests.csv pivot column -> token_type (server_tool_use excluded: never
# priced, not part of the context either).
_REQUEST_TOKEN_FIELDS: tuple[tuple[str, str], ...] = (
    ("input_tokens", "input"),
    ("output_tokens", "output"),
    ("cache_read_tokens", "cache_read"),
    ("cache_write_5m_tokens", "cache_write_5m"),
    ("cache_write_1h_tokens", "cache_write_1h"),
)

# Inter-request gap buckets for the TTL analysis (lo, hi, label), in seconds;
# the first bucket is the incremental-write baseline (no TTL expires within
# 5 minutes), the boundaries match the two cache TTLs.
_GAP_BUCKETS: tuple[tuple[float, float | None, str], ...] = (
    (0, 300, "<= 5m"),
    (300, 3600, "5m-1h"),
    (3600, 21600, "1h-6h"),
    (21600, None, "> 6h"),
)
_BASELINE_GAP = _GAP_BUCKETS[0][2]


def _request_context(row: dict[str, Any]) -> int:
    """Input-side tokens of one request ~= the context window it re-sent."""
    return (
        int(row.get("input_tokens") or 0)
        + int(row.get("cache_read_tokens") or 0)
        + _request_writes(row)
    )


def _request_writes(row: dict[str, Any]) -> int:
    return int(row.get("cache_write_5m_tokens") or 0) + int(row.get("cache_write_1h_tokens") or 0)


def _request_cost(engine: CostEngine, row: dict[str, Any]) -> float:
    model = row.get("model") or ""
    return sum(
        engine.cost(model, token_type, int(row.get(col) or 0))
        for col, token_type in _REQUEST_TOKEN_FIELDS
    )


def _request_write_cost(engine: CostEngine, row: dict[str, Any]) -> float:
    model = row.get("model") or ""
    return engine.cost(
        model, "cache_write_5m", int(row.get("cache_write_5m_tokens") or 0)
    ) + engine.cost(model, "cache_write_1h", int(row.get("cache_write_1h_tokens") or 0))


def _parse_ts(value: str) -> datetime | None:
    """An aware datetime from an ISO timestamp, or None ('Z' included, 3.10)."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _main_chains(ds: Dataset) -> dict[str, list[dict[str, Any]]]:
    """``session_id -> non-sidechain requests`` in chronological order."""
    chains: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds.requests:
        if not row.get("is_sidechain"):
            chains[row["session_id"]].append(row)
    for chain in chains.values():
        chain.sort(key=lambda r: r.get("timestamp") or "")
    return chains


def _percentile(values: list[int], pct: float) -> int:
    """Nearest-rank percentile of a non-empty list of ints."""
    ordered = sorted(values)
    return ordered[round(pct / 100 * (len(ordered) - 1))]


def context_growth(ds: Dataset, provider: str) -> TableResult:
    """Accumulated context per turn by session depth (2.1).

    Each API request re-sends the whole conversation (fresh input + cache
    reads + cache writes). Grouping the main-chain requests by the depth of
    their prompt shows how that context grows as a session deepens: the rent
    every single turn pays. This is the "time to compact" signal the
    per-prompt averages cannot give (06 §3): when the median context stops
    being worth it, /compact or a fresh session resets the curve.
    """
    engine = CostEngine(provider, ds.pricing_path)
    index_of = {row["prompt_id"]: int(row.get("prompt_index") or 0) for row in ds.prompts}

    bands: dict[str, dict[str, Any]] = {
        label: {"contexts": [], "cache_read": 0, "cost": 0.0} for _, _, label in DEPTH_BANDS
    }
    for row in ds.requests:
        if row.get("is_sidechain") or index_of.get(row["prompt_id"], 0) < 1:
            continue
        band = bands[_depth_band(index_of[row["prompt_id"]])]
        band["contexts"].append(_request_context(row))
        band["cache_read"] += int(row.get("cache_read_tokens") or 0)
        band["cost"] += _request_cost(engine, row)

    base_median: int | None = None
    rows: list[dict[str, Any]] = []
    for _, _, label in DEPTH_BANDS:
        contexts: list[int] = bands[label]["contexts"]
        if not contexts:
            continue
        median = int(statistics.median(contexts))
        if base_median is None:
            base_median = median
        n = len(contexts)
        rows.append(
            {
                "depth": label,
                "requests": n,
                "median_context": median,
                "p90_context": _percentile(contexts, 90),
                "cache_read_per_turn": bands[label]["cache_read"] // n,
                "vs_depth_1": round(median / base_median, 2) if base_median else None,
                "cost_per_request_usd": round(bands[label]["cost"] / n, 4),
            }
        )

    notes = [_source_note(ds)]
    if not ds.requests:
        notes.append(_NO_REQUESTS_NOTE)
    notes.append(
        "Context = input-side tokens of one API request (fresh input + cache "
        "read + cache write): what the model re-reads at every turn. One "
        "request ~= one assistant turn; sidechain (subagent) requests are excluded."
    )
    if len(rows) > 1:
        deepest = rows[-1]
        notes.append(
            f"Median context at depth {deepest['depth']}: "
            f"{deepest['median_context']:,} tokens (x{deepest['vs_depth_1']} the "
            "depth-1 median) -- every turn at that depth pays that rent until "
            "/compact or a new session resets it."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Accumulated context by session depth ({provider})",
        [
            Column("depth", "Depth"),
            Column("requests", "Requests", "int"),
            Column("median_context", "Median context", "int"),
            Column("p90_context", "P90 context", "int"),
            Column("cache_read_per_turn", "Cache read/turn", "int"),
            Column("vs_depth_1", "vs depth 1", "x"),
            Column("cost_per_request_usd", "$/request", "money"),
        ],
        rows,
        notes,
    )


def ttl_losses(ds: Dataset, provider: str) -> TableResult:
    """Cache TTL expiry losses (2.2): what pauses cost in cache re-writes.

    Cache entries expire after their TTL (5m or 1h). When work resumes after
    a longer pause, the next request must re-write the expired context -- a
    cache_write spike right after a gap. Each inter-request gap of a
    session's main chain is bucketed by pause length and carries the cache
    writes of the request that follows it. Writes after gaps <= 5m are the
    incremental baseline (nothing expires that fast); whatever exceeds that
    baseline after a longer pause is the estimated expiry loss. The estimate
    is self-correcting: 1h-TTL entries surviving a 30-minute pause simply
    produce no spike, hence no measured loss.
    """
    engine = CostEngine(provider, ds.pricing_path)
    # label -> list of (write_tokens, write_cost) for the request after a gap.
    events: dict[str, list[tuple[int, float]]] = {label: [] for _, _, label in _GAP_BUCKETS}

    def _bucket(gap: float) -> str:
        for lo, hi, label in _GAP_BUCKETS:
            if gap >= lo and (hi is None or gap < hi):
                return label
        return _GAP_BUCKETS[-1][2]

    for chain in _main_chains(ds).values():
        for prev, cur in zip(chain, chain[1:], strict=False):
            t0, t1 = _parse_ts(prev.get("timestamp") or ""), _parse_ts(cur.get("timestamp") or "")
            if t0 is None or t1 is None or t1 < t0:
                continue
            gap = (t1 - t0).total_seconds()
            events[_bucket(gap)].append((_request_writes(cur), _request_write_cost(engine, cur)))

    baseline_writes = [tokens for tokens, _ in events[_BASELINE_GAP]]
    baseline = int(statistics.median(baseline_writes)) if baseline_writes else 0

    def _excess(bucket: list[tuple[int, float]]) -> float:
        """Write cost in excess of the baseline (prorated per event)."""
        return sum(
            cost * (tokens - baseline) / tokens for tokens, cost in bucket if tokens > baseline > 0
        ) + sum(cost for tokens, cost in bucket if tokens > 0 and baseline == 0)

    rows: list[dict[str, Any]] = []
    for _, _, label in _GAP_BUCKETS:
        bucket = events[label]
        if not bucket:
            continue
        total_cost = sum(cost for _, cost in bucket)
        rows.append(
            {
                "gap": label,
                "events": len(bucket),
                "avg_write_tokens": sum(tokens for tokens, _ in bucket) // len(bucket),
                "write_cost_usd": round(total_cost, 4),
                "excess_cost_usd": (None if label == _BASELINE_GAP else round(_excess(bucket), 4)),
            }
        )

    notes = [_source_note(ds)]
    if not ds.requests:
        notes.append(_NO_REQUESTS_NOTE)
    notes.append(
        "Gap = pause between two consecutive main-chain requests of the same "
        "session; the cache writes of the request that follows the pause are "
        f"attributed to it. Baseline incremental write: median {baseline:,} "
        "tokens after gaps <= 5m (nothing expires that fast)."
    )
    long_buckets = [row for row in rows if row["gap"] in ("1h-6h", "> 6h")]
    mid = next((row for row in rows if row["gap"] == "5m-1h"), None)
    if long_buckets:
        loss = sum(row["excess_cost_usd"] for row in long_buckets)
        count = sum(row["events"] for row in long_buckets)
        line = (
            f"Estimated TTL-expiry losses: ${loss:,.2f} of cache re-writes across "
            f"{count} resumptions after pauses > 1h (everything expired)"
        )
        if mid is not None:
            line += (
                f", plus ${mid['excess_cost_usd']:,.2f} after pauses of 5m-1h "
                "(only 5m-TTL entries expire there)"
            )
        notes.append(line + ".")
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Cache TTL expiry losses ({provider})",
        [
            Column("gap", "Pause"),
            Column("events", "Events", "int"),
            Column("avg_write_tokens", "Avg cache write", "int"),
            Column("write_cost_usd", "Write cost", "money"),
            Column("excess_cost_usd", "Est. expiry loss", "money"),
        ],
        rows,
        notes,
    )


def compactions(ds: Dataset, provider: str) -> TableResult:
    """Compaction analysis (2.3): how /compact resets context, at what price.

    A compaction replaces the conversation with a synthetic summary; the
    requests descending from it carry ``post_compact=1`` (1.4). Each 0 -> 1
    transition along a session's main chain is one compaction event: the
    surrounding requests give the context before/after, and the first
    post-compaction request's cache writes are the cost of rebuilding the
    cache from the summary.
    """
    engine = CostEngine(provider, ds.pricing_path)
    session_project = _session_project(ds)

    rows: list[dict[str, Any]] = []
    sessions_hit: set[str] = set()
    for session_id, chain in _main_chains(ds).items():
        prev: dict[str, Any] | None = None
        for row in chain:
            if row.get("post_compact") and not (prev or {}).get("post_compact"):
                before = _request_context(prev) if prev else None
                after = _request_context(row)
                sessions_hit.add(session_id)
                rows.append(
                    {
                        "timestamp": row.get("timestamp") or "",
                        "session_id": session_id,
                        "project": session_project.get(session_id, ""),
                        "context_before": before,
                        "context_after": after,
                        "reduction_pct": (round(100 * (1 - after / before), 1) if before else None),
                        "rebuild_tokens": _request_writes(row),
                        "rebuild_cost_usd": round(_request_write_cost(engine, row), 4),
                    }
                )
            prev = row
    rows.sort(key=lambda r: r["timestamp"])

    notes = [_source_note(ds)]
    if not ds.requests:
        notes.append(_NO_REQUESTS_NOTE)
    if rows:
        befores = [r["context_before"] for r in rows if r["context_before"]]
        afters = [r["context_after"] for r in rows if r["context_before"]]
        if befores:
            median_before = int(statistics.median(befores))
            median_after = int(statistics.median(afters))
            drop = round(100 * (1 - median_after / median_before), 1) if median_before else 0.0
            notes.append(
                f"{len(rows)} compaction(s) across {len(sessions_hit)} of "
                f"{len(ds.sessions)} sessions; median context "
                # Signed so a context that *grew* reads "+x%", never "--x%".
                f"{median_before:,} -> {median_after:,} tokens ({-drop:+.1f}%)."
            )
        total_rebuild = sum(r["rebuild_cost_usd"] for r in rows)
        notes.append(
            f"Rebuilding the cache after compaction (first post-compaction "
            f"request's writes) cost ${total_rebuild:,.2f} in total."
        )
    else:
        notes.append("No compaction found in this history.")
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Compactions ({provider})",
        [
            Column("timestamp", "When (UTC)"),
            Column("session_id", "Session"),
            Column("project", "Project"),
            Column("context_before", "Context before", "int"),
            Column("context_after", "Context after", "int"),
            Column("reduction_pct", "Reduction", "pct"),
            Column("rebuild_tokens", "Rebuild write", "int"),
            Column("rebuild_cost_usd", "Rebuild cost", "money"),
        ],
        rows,
        notes,
    )


def session_overhead(ds: Dataset, provider: str) -> TableResult:
    """Fixed per-session overhead (2.4): system prompt + CLAUDE.md + MCP tools.

    The very first API request of a session carries no conversation yet: its
    input side is the fixed setup every session pays -- system prompt,
    CLAUDE.md, MCP tool definitions -- plus the (usually small) first user
    message. The median across sessions is the displayed estimate; the trend
    over time shows a growing CLAUDE.md or newly added MCP servers.
    """
    engine = CostEngine(provider, ds.pricing_path)
    session_meta = {row["session_id"]: row for row in ds.sessions}
    first_prompts = {
        row["prompt_id"]: row for row in ds.prompts if int(row.get("prompt_index") or 0) == 1
    }

    first_requests: dict[str, dict[str, Any]] = {}
    for row in ds.requests:
        if row.get("is_sidechain") or row["prompt_id"] not in first_prompts:
            continue
        best = first_requests.get(row["session_id"])
        if best is None or int(row["request_index"]) < int(best["request_index"]):
            first_requests[row["session_id"]] = row

    rows: list[dict[str, Any]] = []
    for session_id, row in first_requests.items():
        prompt = first_prompts[row["prompt_id"]]
        rows.append(
            {
                "started": _parse_day(prompt.get("timestamp") or row.get("timestamp") or ""),
                "session_id": session_id,
                "project": session_meta.get(session_id, {}).get("project", ""),
                "model": row.get("model") or "",
                "overhead_tokens": _request_context(row),
                "cache_write_tokens": _request_writes(row),
                "first_request_cost_usd": round(_request_cost(engine, row), 4),
            }
        )
    rows.sort(key=lambda r: (r["started"], r["session_id"]))

    notes = [_source_note(ds)]
    if not ds.requests:
        notes.append(_NO_REQUESTS_NOTE)
    if rows:
        overheads = [r["overhead_tokens"] for r in rows]
        costs = [r["first_request_cost_usd"] for r in rows]
        notes.append(
            "Fixed session overhead (system prompt + CLAUDE.md + MCP tools) "
            f"~= {int(statistics.median(overheads)):,} tokens: the median "
            f"first-turn context across {len(rows)} sessions "
            f"(p10 {_percentile(overheads, 10):,}, p90 {_percentile(overheads, 90):,}). "
            "The first user message is included (usually small)."
        )
        notes.append(
            f"Paying that overhead on turn 1 cost ${sum(costs):,.2f} in total "
            f"(median ${statistics.median(costs):,.2f} per session)."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Fixed session overhead ({provider})",
        [
            Column("started", "Started"),
            Column("session_id", "Session"),
            Column("project", "Project"),
            Column("model", "Model"),
            Column("overhead_tokens", "First-turn context", "int"),
            Column("cache_write_tokens", "Cache write", "int"),
            Column("first_request_cost_usd", "Turn-1 cost", "money"),
        ],
        rows,
        notes,
    )


# ---------------------------------------------------------------------------
# Cross-model what-if (2.5), prescriptive recommendations (2.6), burn rate
# (2.7). These three are the "what to do" layer on top of the descriptive
# analyses above; they re-use the same CostEngine so a re-pricing is always
# consistent with the headline numbers.
# ---------------------------------------------------------------------------


def model_category(ds: Dataset, provider: str, *, target_model: str | None = None) -> TableResult:
    """Cost crossed by model x category, with an optional re-pricing (2.5).

    Each token row is grouped by its model and the category of its prompt
    (``categorize`` output; ``(uncategorized)`` otherwise). That cross alone
    answers "which model am I using for which kind of work, and what does it
    cost". When ``target_model`` is given, every cell is *re-priced* on that
    model with the very same token counts -- the trivial what-if the audit
    asked for (06 §3): "these category-X prompts on Opus would have cost Y on
    Sonnet". server_tool_use is never priced (excluded from costs and tokens).
    """
    engine = CostEngine(provider, ds.pricing_path)
    category_of = {
        pid: (info.get("category") or "(uncategorized)") for pid, info in ds.categories.items()
    }
    real = _real_prompt_ids(ds)

    cost: defaultdict[tuple[str, str], float] = defaultdict(float)
    repriced: defaultdict[tuple[str, str], float] = defaultdict(float)
    tokens: Counter[tuple[str, str]] = Counter()
    prompt_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in ds.tokens:
        model = row.get("model") or "(unknown)"
        key = (model, category_of.get(row["prompt_id"], "(uncategorized)"))
        cost[key] += engine.cost(model, row["token_type"], row["token_count"])
        if target_model:
            repriced[key] += engine.cost(target_model, row["token_type"], row["token_count"])
        if row["token_type"] != "server_tool_use":
            tokens[key] += row["token_count"]
        if row["prompt_id"] in real:
            prompt_sets[key].add(row["prompt_id"])

    total_cost = sum(cost.values())
    rows: list[dict[str, Any]] = []
    for (model, category), cell_cost in sorted(cost.items(), key=lambda kv: -kv[1]):
        key = (model, category)
        out_row: dict[str, Any] = {
            "model": model,
            "category": category,
            "prompts": len(prompt_sets.get(key, set())),
            "tokens": tokens.get(key, 0),
            "cost_usd": round(cell_cost, 4),
            "share_pct": round(100 * cell_cost / total_cost, 1) if total_cost else 0.0,
        }
        if target_model:
            cell_repriced = repriced.get(key, 0.0)
            out_row["repriced_usd"] = round(cell_repriced, 4)
            out_row["saving_usd"] = round(cell_cost - cell_repriced, 4)
        rows.append(out_row)

    columns = [
        Column("model", "Model"),
        Column("category", "Category"),
        Column("prompts", "Prompts", "int"),
        Column("tokens", "Tokens", "int"),
        Column("cost_usd", f"Cost ({provider})", "money"),
        Column("share_pct", "Share", "pct"),
    ]
    title = f"Cost by model x category ({provider})"
    notes = [_source_note(ds)]
    if not ds.categories:
        notes.append(
            "No categorization found -- run `prompt-analytics categorize` so the "
            "category column is more than (uncategorized)."
        )
    if target_model:
        columns += [
            Column("repriced_usd", f"On {target_model}", "money"),
            Column("saving_usd", "Saving", "money"),
        ]
        total_repriced = sum(repriced.values())
        saving = total_cost - total_repriced
        pct = round(100 * saving / total_cost, 1) if total_cost else 0.0
        title = f"What-if: every model re-priced on {target_model} ({provider})"
        notes.append(
            f"Re-pricing all usage on {target_model}: ${total_cost:,.2f} -> "
            f"${total_repriced:,.2f} (a ${saving:,.2f} / {pct}% change at identical "
            "token counts). Cells already on that model are unchanged."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(title, columns, rows, notes)


def _post_compaction_baseline(ds: Dataset) -> int:
    """Estimated context right after a /compact, in tokens.

    Prefers the median observed post-compaction context (2.3); falls back to
    the median first-turn session overhead (2.4) -- both are the size a fresh
    summarized conversation carries -- then to a conservative constant.
    """
    afters: list[int] = []
    for chain in _main_chains(ds).values():
        prev: dict[str, Any] | None = None
        for row in chain:
            if row.get("post_compact") and not (prev or {}).get("post_compact"):
                afters.append(_request_context(row))
            prev = row
    if afters:
        return int(statistics.median(afters))

    first_prompts = {
        row["prompt_id"] for row in ds.prompts if int(row.get("prompt_index") or 0) == 1
    }
    by_session: dict[str, dict[str, Any]] = {}
    for row in ds.requests:
        if row.get("is_sidechain") or row["prompt_id"] not in first_prompts:
            continue
        best = by_session.get(row["session_id"])
        if best is None or int(row["request_index"]) < int(best["request_index"]):
            by_session[row["session_id"]] = row
    firsts = [_request_context(row) for row in by_session.values()]
    if firsts:
        return int(statistics.median(firsts))
    return 20_000


def recommendations(
    ds: Dataset, provider: str, *, min_prompts: int = 50, compact_at: int = 30
) -> TableResult:
    """Prescriptive estimate: what compacting long sessions earlier would save (2.6).

    The first *actionable* output (06 §4). For every session longer than
    ``min_prompts`` real prompts, it reports the cache rent (cache reads +
    writes) it actually paid, then estimates what a /compact around prompt
    ``compact_at`` would have cost instead: past that depth only the context
    *above* a post-compaction baseline is assumed to stay cached, so the cache
    reads beyond that baseline are the modeled saving, minus one cache rebuild
    write. It is deliberately an upper bound on the saving (it ignores the
    context regrowing after the compaction), and it leans on the 2.3/2.4
    baselines so the number stays grounded in the user's own history.
    """
    engine = CostEngine(provider, ds.pricing_path)
    index_of = {row["prompt_id"]: int(row.get("prompt_index") or 0) for row in ds.prompts}
    session_project = _session_project(ds)
    session_prompts: Counter[str] = Counter(row["session_id"] for row in ds.prompts)
    long_sessions = {sid for sid, n in session_prompts.items() if n > min_prompts}
    baseline = _post_compaction_baseline(ds)
    chains = _main_chains(ds)

    rows: list[dict[str, Any]] = []
    for session_id in long_sessions:
        rent = 0.0
        saved_reads = 0.0
        first_post: dict[str, Any] | None = None
        for row in chains.get(session_id, []):
            model = row.get("model") or ""
            read_tokens = int(row.get("cache_read_tokens") or 0)
            rent += engine.cost(model, "cache_read", read_tokens) + _request_write_cost(engine, row)
            if index_of.get(row["prompt_id"], 0) > compact_at:
                saved_reads += engine.cost(model, "cache_read", max(0, read_tokens - baseline))
                if first_post is None:
                    first_post = row
        if first_post is None:
            continue  # the session never reaches the compaction depth
        rebuild = engine.cost(first_post.get("model") or "", "cache_write_1h", baseline)
        net_saving = saved_reads - rebuild
        rows.append(
            {
                "session_id": session_id,
                "project": session_project.get(session_id, ""),
                "prompts": session_prompts[session_id],
                "rent_usd": round(rent, 4),
                "est_compacted_usd": round(rent - net_saving, 4),
                "saving_usd": round(net_saving, 4),
            }
        )
    rows.sort(key=lambda r: -r["saving_usd"])

    notes = [_source_note(ds)]
    if not ds.requests:
        notes.append(_NO_REQUESTS_NOTE)
    if rows:
        total_rent = sum(r["rent_usd"] for r in rows)
        total_compacted = sum(r["est_compacted_usd"] for r in rows)
        total_saving = sum(r["saving_usd"] for r in rows)
        notes.append(
            f"Your {len(rows)} session(s) over {min_prompts} prompts paid "
            f"${total_rent:,.2f} in cache rent; compacting around prompt {compact_at} "
            f"would have cost about ${total_compacted:,.2f} -- an estimated "
            f"${total_saving:,.2f} saved."
        )
        notes.append(
            f"Estimate: post-compaction context ~= {baseline:,} tokens (from your "
            f"compaction history / session overhead); past depth {compact_at} only the "
            "cache reads above that baseline are counted as saved, minus one rebuild "
            "write. Upper bound -- it ignores context regrowth after the compaction."
        )
    else:
        notes.append(
            f"No session exceeds {min_prompts} prompts past depth {compact_at}: nothing "
            "to recommend (your sessions are already short enough)."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Compaction recommendations ({provider})",
        [
            Column("session_id", "Session"),
            Column("project", "Project"),
            Column("prompts", "Prompts", "int"),
            Column("rent_usd", "Cache rent paid", "money"),
            Column("est_compacted_usd", "Est. if compacted", "money"),
            Column("saving_usd", "Est. saving", "money"),
        ],
        rows,
        notes,
    )


def _monday(day: str) -> str:
    """The ISO date of the Monday of ``day``'s week (day = 'YYYY-MM-DD')."""
    dt = datetime.fromisoformat(day)
    return (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")


def burn_rate(ds: Dataset, provider: str, *, weeks: int = 8) -> TableResult:
    """Spend trend: $/day and week-over-week (2.7).

    Each real prompt's cost is attributed to its day, then rolled up into ISO
    weeks (Monday-started). The table lists the most recent ``weeks`` weeks
    with their cost, active days, $/day and the change versus the previous
    week; the notes give the overall burn rate and a last-7-days vs prior-7
    comparison. Session overhead (continuation pseudo-prompts, no timestamp)
    is the only cost left out -- it is small and not attributable to a day.
    """
    engine = CostEngine(provider, ds.pricing_path)
    prompt_costs = _prompt_costs(ds, engine)

    daily_cost: defaultdict[str, float] = defaultdict(float)
    daily_prompts: Counter[str] = Counter()
    for row in ds.prompts:
        day = _parse_day(row.get("timestamp", ""))
        if not day:
            continue
        daily_cost[day] += prompt_costs.get(row["prompt_id"], 0.0)
        daily_prompts[day] += 1

    week_cost: defaultdict[str, float] = defaultdict(float)
    week_prompts: Counter[str] = Counter()
    week_days: defaultdict[str, set[str]] = defaultdict(set)
    for day, cost in daily_cost.items():
        wk = _monday(day)
        week_cost[wk] += cost
        week_days[wk].add(day)
    for day, count in daily_prompts.items():
        week_prompts[_monday(day)] += count

    ordered = sorted(week_cost)
    all_rows: list[dict[str, Any]] = []
    prev_cost: float | None = None
    for wk in ordered:
        cost = week_cost[wk]
        vs_prev = (
            round(100 * (cost - prev_cost) / prev_cost, 1)
            if prev_cost is not None and prev_cost != 0
            else None
        )
        all_rows.append(
            {
                "week_of": wk,
                "active_days": len(week_days[wk]),
                "prompts": week_prompts[wk],
                "cost_usd": round(cost, 4),
                "per_day_usd": round(cost / 7, 4),
                "vs_prev_pct": vs_prev,
            }
        )
        prev_cost = cost
    rows = all_rows[-weeks:] if weeks else all_rows

    notes = [_source_note(ds)]
    if daily_cost:
        days = sorted(daily_cost)
        first, last = datetime.fromisoformat(days[0]), datetime.fromisoformat(days[-1])
        span = (last - first).days + 1
        total = sum(daily_cost.values())
        notes.append(
            f"Burn rate: ${total / span:,.2f}/day over the {span}-day span (total ${total:,.2f})."
        )
        last7 = sum(c for d, c in daily_cost.items() if (last - datetime.fromisoformat(d)).days < 7)
        prior7 = sum(
            c for d, c in daily_cost.items() if 7 <= (last - datetime.fromisoformat(d)).days < 14
        )
        if prior7:
            delta = 100 * (last7 - prior7) / prior7
            notes.append(
                f"Last 7 days ${last7:,.2f} vs prior 7 days ${prior7:,.2f} ({delta:+.0f}%)."
            )
        else:
            notes.append(f"Last 7 days: ${last7:,.2f}.")
    else:
        notes.append("No dated prompts to chart a trend.")
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Weekly burn rate ({provider})",
        [
            Column("week_of", "Week of"),
            Column("active_days", "Active days", "int"),
            Column("prompts", "Prompts", "int"),
            Column("cost_usd", "Cost", "money"),
            Column("per_day_usd", "$/day", "money"),
            Column("vs_prev_pct", "vs prev week", "pct"),
        ],
        rows,
        notes,
    )


TIMELINE_PERIODS: tuple[str, ...] = ("day", "week", "month")

_PERIOD_LABEL = {"day": "Day", "week": "Week of", "month": "Month"}


def _period_bucket(day: str, by: str) -> str:
    """Bucket a ``YYYY-MM-DD`` day into its day / ISO-week-Monday / month key."""
    if by == "week":
        return _monday(day)
    if by == "month":
        return day[:7]
    return day


def timeline(ds: Dataset, provider: str, *, by: str = "day") -> TableResult:
    """Cost, prompts and tokens grouped by calendar period.

    ``by`` is one of ``day`` / ``week`` / ``month``. Each real prompt's cost and
    tokens are attributed to the period of its ``timestamp`` and summed; periods
    are listed chronologically, each with its share of the total cost. Like
    ``burn-rate``, undated usage (continuation pseudo-prompts) is left out -- it
    is small and not attributable to a day.
    """
    if by not in TIMELINE_PERIODS:
        raise ValueError(f"Unknown period {by!r}; expected one of {', '.join(TIMELINE_PERIODS)}.")
    engine = CostEngine(provider, ds.pricing_path)
    prompt_costs = _prompt_costs(ds, engine)
    token_counts = _prompt_token_counts(ds)

    cost: defaultdict[str, float] = defaultdict(float)
    prompts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for row in ds.prompts:
        day = _parse_day(row.get("timestamp", ""))
        if not day:
            continue
        bucket = _period_bucket(day, by)
        pid = row["prompt_id"]
        cost[bucket] += prompt_costs.get(pid, 0.0)
        prompts[bucket] += 1
        tokens[bucket] += _token_total(token_counts.get(pid, Counter()))

    total_cost = sum(cost.values())
    rows: list[dict[str, Any]] = [
        {
            "period": bucket,
            "prompts": prompts[bucket],
            "tokens": tokens[bucket],
            "cost_usd": round(cost[bucket], 4),
            "share_pct": round(100 * cost[bucket] / total_cost, 1) if total_cost else 0.0,
        }
        for bucket in sorted(cost)
    ]

    notes = [_source_note(ds)]
    if not rows:
        notes.append("No dated prompts to group.")
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)
    return TableResult(
        f"Cost by {by}",
        [
            Column("period", _PERIOD_LABEL[by]),
            Column("prompts", "Prompts", "int"),
            Column("tokens", "Tokens", "int"),
            Column("cost_usd", f"Cost ({provider})", "money"),
            Column("share_pct", "Cost %", "pct"),
        ],
        rows,
        notes,
    )


def _span_days(ds: Dataset) -> int:
    """Number of calendar days the dated prompts span (>= 1)."""
    days = sorted({_parse_day(p.get("timestamp", "")) for p in ds.prompts if p.get("timestamp")})
    if not days:
        return 1
    span = (datetime.fromisoformat(days[-1]) - datetime.fromisoformat(days[0])).days + 1
    return max(span, 1)


def break_even(
    ds: Dataset,
    *,
    provider: str = "anthropic",
    quota_rows: list[dict[str, Any]] | None = None,
) -> TableResult:
    """Plan break-even: API-equivalent value vs flat-rate subscription (3.1).

    THE question the cost audit left unanswered (06 §3, "Plan Max vs API"). It
    prices the whole history on the per-token ``provider`` grid -- the
    *API-equivalent* of the usage -- projects it to a month over the observed
    span, and compares that against each ``plans:`` subscription: "your usage is
    worth $X of API for $Y/month of subscription". A plan pays off once the
    monthly API-equivalent exceeds its price.

    Fallback (required): the break-even stands on the API-equivalent alone, so
    it works with no ``quota_log.csv`` at all. When quota snapshots exist they
    enrich the notes (peak utilization of each plan window) but are never
    required -- ``snapshot`` only runs on demand. (06 §5.10)
    """
    from .pricing import load_plans

    engine = CostEngine(provider, ds.pricing_path)
    api_value = sum(_prompt_costs(ds, engine).values())
    span = _span_days(ds)
    monthly_api = api_value / span * 30

    plans = load_plans(ds.pricing_path)
    rows: list[dict[str, Any]] = []
    for name, plan in plans.items():
        price = float(plan.get("monthly_usd", 0.0))
        rows.append(
            {
                "plan": plan.get("label") or name,
                "monthly_price_usd": round(price, 2),
                "api_equiv_month_usd": round(monthly_api, 2),
                "vs_plan": round(monthly_api / price, 2) if price else None,
                "saving_month_usd": round(monthly_api - price, 2),
            }
        )
    rows.sort(key=lambda r: r["monthly_price_usd"])

    notes = [_source_note(ds)]
    if not plans:
        notes.append(
            "No subscription plans in pricing.yml -- add a `plans:` section "
            "(label + monthly_usd) to compare against the API-equivalent."
        )
    notes.append(
        f"Over the {span}-day window your usage is worth ${api_value:,.2f} of {provider} API "
        f"(${monthly_api:,.2f}/month projected at this rate)."
    )
    worth_it = [r for r in rows if r["saving_month_usd"] > 0]
    if worth_it:
        best = max(worth_it, key=lambda r: r["saving_month_usd"])
        notes.append(
            f"At this rate the {best['plan']} plan (${best['monthly_price_usd']:,.2f}/mo) pays for "
            f"itself: your API-equivalent is ${best['api_equiv_month_usd']:,.2f}/month "
            f"(x{best['vs_plan']} the price), an estimated ${best['saving_month_usd']:,.2f}/month "
            "cheaper than paying per token."
        )
    elif rows:
        cheapest = rows[0]
        notes.append(
            f"At this rate no plan pays off: even the {cheapest['plan']} plan "
            f"(${cheapest['monthly_price_usd']:,.2f}/mo) costs more than your "
            f"${cheapest['api_equiv_month_usd']:,.2f}/month of API-equivalent -- pay-as-you-go API "
            "is cheaper for this volume."
        )

    # Quota enrichment (optional): peak utilization per plan window, latest first.
    peaks: dict[str, float] = {}
    for row in quota_rows or []:
        field = str(row.get("field") or "")
        raw = row.get("utilization_pct")
        if raw is None or raw == "":
            continue
        try:
            util = float(raw)
        except (TypeError, ValueError):
            continue
        if field:
            peaks[field] = max(peaks.get(field, 0.0), util)
    if peaks:
        parts = ", ".join(f"{field} {peak:.0f}%" for field, peak in sorted(peaks.items()))
        notes.append(
            f"Quota windows (peak utilization seen via `snapshot`): {parts}. "
            "High utilization means you are extracting most of the plan's allowance; "
            "low utilization means headroom (or a smaller plan would do)."
        )
    else:
        notes.append(
            "No quota snapshots yet: break-even shown from API-equivalent vs plan price only. "
            "Run `prompt-analytics snapshot` to enrich it with how much of each plan window "
            "you actually use."
        )
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)

    return TableResult(
        f"Plan break-even: API-equivalent vs subscription ({provider})",
        [
            Column("plan", "Plan"),
            Column("monthly_price_usd", "Plan $/mo", "money"),
            Column("api_equiv_month_usd", "Your API $/mo", "money"),
            Column("vs_plan", "vs plan", "x"),
            Column("saving_month_usd", "Saving $/mo", "money"),
        ],
        rows,
        notes,
    )


def observed_span_days(ds: Dataset) -> int:
    """Calendar days the dated prompts span (>= 1) -- the window of the data."""
    return _span_days(ds)


def monthly_api_equivalent(ds: Dataset, provider: str = "anthropic") -> float:
    """This usage priced on ``provider``'s per-token grid, projected to a month.

    The same figure :func:`break_even` compares against subscription plans: the
    whole history priced per token, divided by the observed span, times 30.
    """
    engine = CostEngine(provider, ds.pricing_path)
    api_value = sum(_prompt_costs(ds, engine).values())
    return api_value / _span_days(ds) * 30


def copilot_channel_costs(ds: Dataset) -> list[dict[str, Any]]:
    """Effective monthly cost of this usage on each GitHub Copilot tier.

    GitHub Copilot is usage-based (GitHub AI Credits, 1 credit = $0.01): a tier
    bundles ``included_usd`` of usage, and anything beyond it is per-token overage
    on the ``copilot`` grid. So the effective monthly cost is
    ``monthly_usd + max(0, monthly_copilot_api - included_usd)``. Rows are sorted
    cheapest first; empty when the pricing file has no ``copilot_plans`` section
    or no ``copilot`` provider grid.
    """
    from .pricing import load_copilot_plans, load_pricing

    tiers = load_copilot_plans(ds.pricing_path)
    if not tiers or "copilot" not in load_pricing(ds.pricing_path).get("providers", {}):
        return []
    monthly = monthly_api_equivalent(ds, "copilot")
    span = _span_days(ds)
    actual = monthly / 30 * span  # exact inverse of the monthly projection
    rows: list[dict[str, Any]] = []
    for key, tier in tiers.items():
        price = float(tier.get("monthly_usd", 0.0))
        included = float(tier.get("included_usd", 0.0))
        overage = max(0.0, monthly - included)
        rows.append(
            {
                "key": key,
                "label": tier.get("label") or key,
                "monthly_usd": round(price, 2),
                "included_usd": round(included, 2),
                "overage_usd": round(overage, 2),
                "total_usd": round(price + overage, 2),
                "usage_month_usd": round(monthly, 2),
                "usage_actual_usd": round(actual, 2),
                "span_days": span,
            }
        )
    rows.sort(key=lambda r: r["total_usd"])
    return rows


def compare_providers(ds: Dataset, providers: list[str]) -> TableResult:
    """The same usage priced on several provider grids, per model."""
    engines = {provider: CostEngine(provider, ds.pricing_path) for provider in providers}

    tokens: Counter[str] = Counter()
    costs: dict[str, defaultdict[str, float]] = {
        provider: defaultdict(float) for provider in providers
    }
    for row in ds.tokens:
        model = row.get("model") or "(unknown)"
        if row["token_type"] != "server_tool_use":
            tokens[model] += row["token_count"]
        for provider, engine in engines.items():
            costs[provider][model] += engine.cost(
                row.get("model") or "", row["token_type"], row["token_count"]
            )

    first = providers[0]
    models = sorted(tokens, key=lambda m: -costs[first][m])
    rows: list[dict[str, Any]] = []
    for model in models:
        out_row: dict[str, Any] = {"model": model, "tokens": tokens[model]}
        for provider in providers:
            out_row[f"cost_{provider}_usd"] = round(costs[provider][model], 4)
        rows.append(out_row)
    total_row: dict[str, Any] = {"model": "TOTAL", "tokens": sum(tokens.values())}
    for provider in providers:
        total_row[f"cost_{provider}_usd"] = round(sum(costs[provider].values()), 4)
    rows.append(total_row)

    notes = [_source_note(ds)]
    base_total = total_row[f"cost_{first}_usd"]
    for provider in providers[1:]:
        other = total_row[f"cost_{provider}_usd"]
        if base_total:
            notes.append(f"Total on {provider}: x{other / base_total:.2f} the {first} total.")
    notes.append(
        "These are per-token API prices: what you would pay billing this usage through each "
        "provider's API. On a flat-rate plan (Pro / Max) you pay the subscription, not per token "
        "-- run `break-even` to see whether a plan is cheaper than this API-equivalent cost."
    )
    for engine in engines.values():
        if (note := engine.note()) is not None:
            notes.append(note)
        if (lc_note := engine.long_context_note()) is not None:
            notes.append(lc_note)

    columns = [Column("model", "Model"), Column("tokens", "Tokens", "int")]
    columns += [
        Column(f"cost_{provider}_usd", f"Cost ({provider})", "money") for provider in providers
    ]
    return TableResult("Provider cost comparison", columns, rows, notes)


# ---------------------------------------------------------------------------
# Axe E: before/after impact of an optimization (capstone, transverse).
#
# A date pivot splits the history; the SAME workload-normalized ratios are
# computed on both sides so a config change (a CLAUDE.md edit, /compact earlier,
# a model switch) reads as a delta instead of a raw-total swing that the workload
# alone would confound. The confounders (volume, depth, task mix) are surfaced
# right next to the ratios so the change is never over-sold as causal: this is an
# observational split, not a controlled experiment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactMetric:
    """One before/after metric of the Axe-E split (shared shape, CLI + dashboard).

    ``before``/``after`` are raw values (a float ratio, a money amount, a count,
    or a label) or ``None`` when a side has no data; ``fmt`` drives how
    :func:`impact` renders them and their delta. ``confounder`` flags the workload
    descriptors (volume, depth, task mix) shown to keep the ratio deltas honest --
    they describe how the workload moved, they are not the optimization.
    """

    label: str
    before: float | str | None
    after: float | str | None
    fmt: str
    confounder: bool = False


@dataclass(frozen=True)
class ImpactReport:
    """Structured Axe-E before/after report (shared by the CLI + the dashboard).

    ``pivot`` is the inclusive split day: ``before`` covers prompts up to the day
    before it, ``after`` covers the pivot day onward. ``metrics`` lists the
    workload-normalized ratios first, then the confounders. The pure numbers feed
    both :func:`impact` (the CLI table) and the dashboard, so the two never drift.
    """

    provider: str
    pivot: str
    before_prompts: int
    after_prompts: int
    before_days: int
    after_days: int
    metrics: list[ImpactMetric]

    @property
    def has_both_sides(self) -> bool:
        """True when both sides carry at least one prompt (a meaningful split)."""
        return self.before_prompts > 0 and self.after_prompts > 0


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator``, or ``None`` when the denominator is 0."""
    return numerator / denominator if denominator else None


def _impact_side_stats(side: Dataset, provider: str) -> dict[str, Any]:
    """Workload-normalized ratios + confounders for one side of the split.

    Every ratio is normalized (per prompt, per turn, or as a cost share) so it
    isolates the config from the workload; the raw confounders (prompts, cost,
    days, depth, tasks, dominant category) ride alongside so the deltas stay
    honest. A turn is one API request (``requests.csv`` grain).
    """
    engine = CostEngine(provider, side.pricing_path)
    prompts = len(side.prompts)
    sessions = len(side.sessions)

    total_cost = output_cost = input_cost = context_cost_total = 0.0
    cache_read_tokens = output_tokens = 0
    for row in side.tokens:
        token_type = row["token_type"]
        c = engine.cost(row.get("model") or "", token_type, row["token_count"])
        total_cost += c
        if token_type == "output":
            output_cost += c
            output_tokens += row["token_count"]
        elif token_type == "input":
            input_cost += c
        elif token_type in _CONTEXT_TOKEN_TYPES:
            context_cost_total += c
            if token_type == "cache_read":
                cache_read_tokens += row["token_count"]

    turns = len(side.requests)
    categories: Counter[str] = Counter()
    for row in side.prompts:
        category = (side.categories.get(row["prompt_id"]) or {}).get("category") or ""
        if category:
            categories[category] += 1

    return {
        "prompts": prompts,
        "total_cost": round(total_cost, 4),
        "cost_per_prompt": _safe_ratio(total_cost, prompts),
        "output_share": (100 * output_cost / total_cost) if total_cost else None,
        "context_share": (100 * context_cost_total / total_cost) if total_cost else None,
        "cache_read_per_turn": _safe_ratio(cache_read_tokens, turns),
        "output_per_prompt": _safe_ratio(output_tokens, prompts),
        "days": _span_days(side) if prompts else 0,
        "prompts_per_session": _safe_ratio(prompts, sessions),
        "tasks": len(side.tasks),
        "top_category": categories.most_common(1)[0][0] if categories else "(uncategorized)",
    }


def day_before(day: str) -> str:
    """The calendar day before an ISO ``YYYY-MM-DD`` (lexical, no timezone)."""
    return (datetime.fromisoformat(day) - timedelta(days=1)).date().isoformat()


def split_on_pivot(ds: Dataset, pivot: str) -> tuple[Dataset, Dataset]:
    """Split ``ds`` on a pivot day into ``(before, after)`` views (Axe E / DASH2).

    ``before`` covers prompts up to the day before ``pivot``; ``after`` covers the
    pivot day onward -- the same inclusive convention :func:`impact_report` is
    built on (it calls this). Shared with the dashboard's global date-pivot mode so
    the CLI table and the before/after views split the history identically.
    """
    return filter_dates(ds, None, day_before(pivot)), filter_dates(ds, pivot, None)


def impact_report(ds: Dataset, *, provider: str, pivot: str) -> ImpactReport:
    """Assemble the Axe-E before/after report around ``pivot`` (see :class:`ImpactReport`).

    Splits the dataset on the pivot day (``before`` = up to the day before it,
    ``after`` = the pivot day onward, via :func:`split_on_pivot`), then computes
    the same workload-normalized ratios and workload confounders on each side.
    Pure: the filesystem pivot suggestions live in :func:`suggest_pivots`, kept out
    of here so this stays deterministic and unit-testable.
    """
    before, after = split_on_pivot(ds, pivot)
    b = _impact_side_stats(before, provider)
    a = _impact_side_stats(after, provider)

    metrics: list[ImpactMetric] = [
        ImpactMetric("Cost per prompt", b["cost_per_prompt"], a["cost_per_prompt"], "money"),
        ImpactMetric("Output cost share", b["output_share"], a["output_share"], "pct"),
        ImpactMetric("Context rent share", b["context_share"], a["context_share"], "pct"),
        ImpactMetric(
            "Cache read / turn", b["cache_read_per_turn"], a["cache_read_per_turn"], "ratio"
        ),
        ImpactMetric(
            "Output tokens / prompt", b["output_per_prompt"], a["output_per_prompt"], "ratio"
        ),
        ImpactMetric("Prompts", b["prompts"], a["prompts"], "int", confounder=True),
        ImpactMetric(
            "Total cost", b["total_cost"], a["total_cost"], "money_total", confounder=True
        ),
        ImpactMetric("Active days", b["days"], a["days"], "int", confounder=True),
        ImpactMetric(
            "Prompts / session",
            b["prompts_per_session"],
            a["prompts_per_session"],
            "ratio",
            confounder=True,
        ),
    ]
    if ds.tasks:
        metrics.append(ImpactMetric("Tasks", b["tasks"], a["tasks"], "int", confounder=True))
    if ds.categories:
        metrics.append(
            ImpactMetric(
                "Top category", b["top_category"], a["top_category"], "str", confounder=True
            )
        )

    return ImpactReport(
        provider=provider,
        pivot=pivot,
        before_prompts=b["prompts"],
        after_prompts=a["prompts"],
        before_days=b["days"],
        after_days=a["days"],
        metrics=metrics,
    )


def impact_fmt_value(value: float | str | None, fmt: str) -> str:
    """Render a before/after cell (a side may be empty -> ``-``)."""
    if value is None or value == "":
        return "-"
    if fmt == "money":
        return f"${float(value):,.4f}"
    if fmt == "money_total":
        return f"${float(value):,.2f}"
    if fmt == "pct":
        return f"{float(value):.1f}%"
    if fmt == "ratio":
        return f"{float(value):,.1f}"
    if fmt == "int":
        return f"{int(value):,}"
    return str(value)


def impact_fmt_change(before: float | str | None, after: float | str | None, fmt: str) -> str:
    """Render the delta cell: percentage points for shares, signed delta (+ % change)
    for amounts and ratios, a same/changed flag for a label."""
    if fmt == "str":
        if before is None and after is None:
            return "-"
        return "same" if before == after else "changed"
    if before is None or after is None:
        return "-"  # one side empty -> no meaningful delta
    bf, af = float(before), float(after)
    delta = af - bf
    if fmt == "pct":
        # Shares move in percentage POINTS, not a relative percentage.
        return f"{delta:+.1f} pp"
    if fmt == "money":
        body = f"{'+' if delta >= 0 else '-'}${abs(delta):,.4f}"
    elif fmt == "money_total":
        body = f"{'+' if delta >= 0 else '-'}${abs(delta):,.2f}"
    elif fmt == "int":
        body = f"{delta:+,.0f}"
    else:  # ratio
        body = f"{delta:+,.1f}"
    if bf:
        body += f" ({100 * delta / bf:+.0f}%)"
    return body


def impact(
    ds: Dataset,
    provider: str,
    *,
    pivot: str,
    suggestions: list[tuple[str, str]] | None = None,
) -> TableResult:
    """Before/after impact of an optimization around a date ``pivot`` (Axe E).

    Splits the history on ``pivot`` and shows, per metric, the BEFORE value, the
    AFTER value and the change. The headline rows are workload-normalized ratios
    (cost per prompt, output cost share, context rent share, cache read per turn,
    output tokens per prompt) so a config change reads through the workload; the
    rows under the divider are the workload confounders (volume, depth, task mix)
    that keep the deltas honest. The notes spell out that this is an observational
    split, not a controlled experiment. The pure numbers live in
    :func:`impact_report` (shared with the dashboard); ``suggestions`` (from
    :func:`suggest_pivots`) are folded into the notes as a pivot-picking aid.
    """
    report = impact_report(ds, provider=provider, pivot=pivot)

    def _row(metric: ImpactMetric) -> dict[str, Any]:
        return {
            "metric": metric.label,
            "before": impact_fmt_value(metric.before, metric.fmt),
            "after": impact_fmt_value(metric.after, metric.fmt),
            "change": impact_fmt_change(metric.before, metric.after, metric.fmt),
        }

    rows: list[dict[str, Any]] = [_row(m) for m in report.metrics if not m.confounder]
    confounders = [m for m in report.metrics if m.confounder]
    if confounders:
        rows.append(
            {
                "metric": "— workload confounders (not the optimization) —",
                "before": "",
                "after": "",
                "change": "",
            }
        )
        rows.extend(_row(m) for m in confounders)

    pivot_before = day_before(pivot)
    notes = [_source_note(ds)]
    notes.append(
        f"Pivot {pivot}: BEFORE = {report.before_prompts:,} prompts up to {pivot_before} "
        f"({report.before_days} active days); AFTER = {report.after_prompts:,} prompts from "
        f"{pivot} ({report.after_days} active days)."
    )
    if not report.has_both_sides:
        empty = "before" if report.before_prompts == 0 else "after"
        notes.append(
            f"WARNING: no prompts {empty} the pivot -- pick a date inside the data range "
            "for a meaningful comparison."
        )
    notes.append(
        "The ratios above the divider are workload-normalized (per prompt, per turn, or as a "
        "cost share); the confounders below describe how the workload itself moved. This is an "
        "observational split, not a controlled experiment -- if volume, depth or task mix "
        "shifted a lot, read the ratio deltas with caution (correlation, not proven causation)."
    )
    notes.append("Cache read / turn = average context re-sent per API request (a 'turn').")
    if suggestions:
        listed = ", ".join(f"{day} ({label})" for day, label in suggestions)
        notes.append(f"Detected config changes you could use as a pivot: {listed}.")
    # Loud about unpriced / long-context models, like every other cost view.
    engine = CostEngine(provider, ds.pricing_path)
    for row in ds.tokens:
        engine.cost(row.get("model") or "", row["token_type"], row["token_count"])
    if (note := engine.note()) is not None:
        notes.append(note)
    if (lc_note := engine.long_context_note()) is not None:
        notes.append(lc_note)

    return TableResult(
        f"Impact before/after {pivot} ({provider})",
        [
            Column("metric", "Metric"),
            Column("before", "Before", "str"),
            Column("after", "After", "str"),
            Column("change", "Change", "str"),
        ],
        rows,
        notes,
    )


def suggest_pivots(ds: Dataset) -> list[tuple[str, str]]:
    """Candidate pivot dates from the mtime of config files (Axe E pivot aid).

    Probes the Claude config dir (``CLAUDE.md`` / ``settings.json`` /
    ``settings.local.json``) and every project ``cwd`` seen in the data for a
    ``CLAUDE.md`` or ``.claude/settings*.json``; a file's last-modified day is a
    plausible "I changed my setup here" pivot. This is only a typing aid -- it
    never drives the analysis -- and is filtered to dates strictly inside the data
    range so the suggested split is non-degenerate. Returns ``(YYYY-MM-DD, label)``
    pairs sorted by day, deduplicated.
    """
    days = sorted(_parse_day(p.get("timestamp", "")) for p in ds.prompts if p.get("timestamp"))
    if not days:
        return []
    first, last = days[0], days[-1]

    candidates: list[tuple[Path, str]] = []
    config_dir = paths.claude_config_dir()
    for name in ("CLAUDE.md", "settings.json", "settings.local.json"):
        candidates.append((config_dir / name, f"~/.claude/{name}"))
    cwds = {row.get("cwd") or "" for row in ds.sessions} | {
        row.get("cwd") or "" for row in ds.prompts
    }
    for cwd in sorted(c for c in cwds if c):
        base = Path(cwd)
        label_base = base.name or cwd
        candidates.append((base / "CLAUDE.md", f"{label_base}/CLAUDE.md"))
        candidates.append(
            (base / ".claude" / "settings.json", f"{label_base}/.claude/settings.json")
        )
        candidates.append(
            (base / ".claude" / "settings.local.json", f"{label_base}/.claude/settings.local.json")
        )

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for path, label in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        day = datetime.fromtimestamp(mtime).date().isoformat()
        if not (first < day <= last):
            continue
        key = (day, label)
        if key not in seen:
            seen.add(key)
            out.append(key)
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Flat export (7.5) and the post-extract mini summary (7.6).
# ---------------------------------------------------------------------------


def flat_export(
    ds: Dataset, providers: list[str] | None = None
) -> tuple[list[str], list[dict[str, Any]]]:
    """One denormalized row per prompt for Excel/BI (``export --flat``).

    Token counts are pivoted into columns, per-provider costs are computed,
    and session info is duplicated onto every row. Pseudo-prompts (session
    overhead such as continuation tails) are included with an empty
    ``prompt_index`` so that cost totals reconcile with ``summary``.
    """
    providers = providers or known_providers(ds.pricing_path)
    engines = {provider: CostEngine(provider, ds.pricing_path) for provider in providers}
    token_counts = _prompt_token_counts(ds)
    prompt_models: dict[str, set[str]] = defaultdict(set)
    prompt_session: dict[str, str] = {}
    subagent_tokens: Counter[str] = Counter()
    for row in ds.tokens:
        if row.get("model"):
            prompt_models[row["prompt_id"]].add(row["model"])
        prompt_session.setdefault(row["prompt_id"], row["session_id"])
        if row.get("is_sidechain") and row["token_type"] != "server_tool_use":
            subagent_tokens[row["prompt_id"]] += row["token_count"]
    session_meta = {row["session_id"]: row for row in ds.sessions}

    cost_cols = [f"cost_{provider}_usd" for provider in providers]
    columns = [
        "session_id",
        "session_start_date",
        "project",
        "git_branch",
        "prompt_id",
        "prompt_index",
        "timestamp",
        "model",
        "category",
        "complexity",
        "char_count",
        "assistant_turns",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_5m_tokens",
        "cache_write_1h_tokens",
        "server_tool_use_requests",
        "total_tokens",
        "subagent_tokens",
        *cost_cols,
        "prompt_preview",
    ]

    token_rows_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trow in ds.tokens:
        token_rows_by_prompt[trow["prompt_id"]].append(trow)

    def _costs_of(pid: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for provider, engine in engines.items():
            total = 0.0
            for trow in token_rows_by_prompt.get(pid, []):
                total += engine.cost(
                    trow.get("model") or "", trow["token_type"], trow["token_count"]
                )
            out[f"cost_{provider}_usd"] = round(total, 6)
        return out

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for prow in ds.prompts:
        pid = prow["prompt_id"]
        seen.add(pid)
        counts = token_counts.get(pid, Counter())
        session = session_meta.get(prow["session_id"], {})
        info = ds.categories.get(pid, {})
        rows.append(
            {
                "session_id": prow["session_id"],
                "session_start_date": session.get("start_date", ""),
                "project": prow.get("project", ""),
                "git_branch": prow.get("git_branch", ""),
                "prompt_id": pid,
                "prompt_index": prow.get("prompt_index", ""),
                "timestamp": prow.get("timestamp", ""),
                "model": prow.get("model", ""),
                "category": info.get("category", ""),
                "complexity": info.get("complexity", ""),
                "char_count": prow.get("char_count", ""),
                "assistant_turns": prow.get("assistant_turns", ""),
                "tool_calls": prow.get("tool_calls", ""),
                "input_tokens": counts.get("input", 0),
                "output_tokens": counts.get("output", 0),
                "cache_read_tokens": counts.get("cache_read", 0),
                "cache_write_5m_tokens": counts.get("cache_write_5m", 0),
                "cache_write_1h_tokens": counts.get("cache_write_1h", 0),
                "server_tool_use_requests": counts.get("server_tool_use", 0),
                "total_tokens": _token_total(counts),
                "subagent_tokens": subagent_tokens.get(pid, 0),
                **_costs_of(pid),
                "prompt_preview": prow.get("prompt_preview", ""),
            }
        )

    # Session overhead rows (pseudo-prompts), so totals reconcile with summary.
    for pid in sorted(token_rows_by_prompt):
        if pid in seen:
            continue
        counts = token_counts.get(pid, Counter())
        session_id = prompt_session.get(pid, "")
        session = session_meta.get(session_id, {})
        models = prompt_models.get(pid, set()) - {_SYNTHETIC_MODEL}
        rows.append(
            {
                "session_id": session_id,
                "session_start_date": session.get("start_date", ""),
                "project": session.get("project", ""),
                "git_branch": session.get("git_branch", ""),
                "prompt_id": pid,
                "prompt_index": "",
                "timestamp": "",
                "model": ", ".join(sorted(models)),
                "category": "",
                "complexity": "",
                "char_count": "",
                "assistant_turns": "",
                "tool_calls": "",
                "input_tokens": counts.get("input", 0),
                "output_tokens": counts.get("output", 0),
                "cache_read_tokens": counts.get("cache_read", 0),
                "cache_write_5m_tokens": counts.get("cache_write_5m", 0),
                "cache_write_1h_tokens": counts.get("cache_write_1h", 0),
                "server_tool_use_requests": counts.get("server_tool_use", 0),
                "total_tokens": _token_total(counts),
                "subagent_tokens": subagent_tokens.get(pid, 0),
                **_costs_of(pid),
                "prompt_preview": "",
            }
        )
    return columns, rows


def mini_summary(ds: Dataset, providers: list[str] | None = None) -> list[str]:
    """Two terminal lines for the end of ``extract``/``run`` (7.6)."""
    providers = providers or known_providers(ds.pricing_path)
    parts: list[str] = []
    anthropic_engine: CostEngine | None = None
    for provider in providers:
        engine = CostEngine(provider, ds.pricing_path)
        total = sum(_prompt_costs(ds, engine).values())
        parts.append(f"{provider} ${total:,.2f}")
        if anthropic_engine is None:
            anthropic_engine = engine
    lines = [f"Cost:            {'  |  '.join(parts)}"]

    if anthropic_engine is not None:
        result = by_project(ds, anthropic_engine.provider)
        top = [f"{row['project']} ${row['cost_usd']:,.2f}" for row in result.rows[:3]]
        if top:
            lines.append(f"Top projects:    {', '.join(top)}")
    return lines
