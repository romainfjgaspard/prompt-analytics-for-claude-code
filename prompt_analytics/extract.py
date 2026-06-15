"""Extraction of Claude Code usage data from local JSONL files.

Counting rules (the "How the numbers stay accurate" contract):

* **Global deduplication** -- assistant usage is deduplicated across *all*
  files of the run by ``message.id + requestId`` (resumed/forked sessions
  replay the same records into new files; counting them per-file would double
  the totals). The lines of a message carry *progressive* usage snapshots, so
  the largest snapshot per key wins (ties: the first line). This matches
  ccusage's totals exactly (validated by ``scripts/reconcile_ccusage.py``).
* **Attribution by promptId** -- usage is attached to the prompt referenced by
  the event (its own ``promptId`` in older formats, or the
  ``uuid``/``parentUuid`` chain walked back to a user prompt in newer ones),
  never to "the last prompt seen". Unattributable usage (continuation tails)
  goes to a per-session ``:_continuation`` pseudo prompt: present in
  ``tokens.csv``, absent from ``prompts.csv``.
* **Fake prompts are filtered** -- ``isMeta`` entries, local command echoes
  (``<command-name>``/``<local-command-stdout>``), user interruptions and the
  synthetic post-compaction continuation message never become prompt rows,
  but their token usage is still counted (attached to the session).
* **Sidechains/subagents are included** -- both inline ``isSidechain`` events
  and separate ``subagents/*.jsonl`` files are parsed; their cost is attached
  to the parent prompt (or parent session), and they are excluded from
  ``assistant_turns``/``tool_calls``.
* **Raw counts only** -- no prices are baked into the CSVs (costs are computed
  at read time from the pricing tables). Cache writes are split by TTL
  (``cache_write_5m`` / ``cache_write_1h``) because they are billed
  differently.
* **Request grain is always written** (V10) -- ``requests.csv`` keeps one row
  per deduplicated API request (timestamps, per-request token counts,
  ``is_sidechain``, ``post_compact``); per prompt its sums equal the
  ``tokens.csv`` totals exactly (V7).

Every run regenerates all CSVs atomically (no incremental mode). Performance
on large histories comes from a per-file parse cache keyed by path + mtime +
size (under ``%LOCALAPPDATA%`` / ``$XDG_CACHE_HOME``, see
:mod:`prompt_analytics.paths`), which is always correct: a modified file is
re-parsed in full, and entries of deleted files are garbage-collected.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, tzinfo
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast
from zoneinfo import ZoneInfo

from . import paths
from .pricing import get_model_pricing
from .schema import (
    PROMPT_TEXT_COLS,
    PROMPTS_COLS,
    REQUESTS_COLS,
    SESSIONS_COLS,
    TOKEN_TYPE_DESCRIPTIONS,
    TOKEN_TYPE_LABELS,
    TOKEN_TYPES,
    TOKEN_TYPES_COLS,
    TOKENS_COLS,
    ParsedFile,
    ParsedPrompt,
    PromptRow,
    RequestRow,
    SessionRow,
    TokenRow,
    UsageRecord,
    continuation_prompt_id,
)
from .storage import atomic_write_csv, atomic_write_json

__all__ = ["run_extract", "collect", "parse_file", "ExtractReport", "ExtractResult"]

# Bump whenever parse_file's output shape or logic changes: invalidates the
# on-disk parse cache.
CACHE_VERSION = 2

PREVIEW_CHARS = 100

# Event types we knowingly handle or ignore. Anything else is surfaced in the
# extraction report as a canary for format changes.
KNOWN_EVENT_TYPES = frozenset(
    {
        "user",
        "assistant",
        "mode",
        "system",
        "summary",
        "attachment",
        "ai-title",
        "last-prompt",
        "file-history-snapshot",
        "permission-mode",
        "queue-operation",
        "bridge-session",
    }
)

# Markers of "fake" user prompts (3.5).
_COMMAND_TAGS = ("<command-name>", "<command-message>", "<local-command-stdout>")
_CONTINUATION_PREFIX = "This session is being continued from a previous conversation"
_INTERRUPT_PREFIX = "[Request interrupted by user"

# Pseudo model id used by Claude Code for synthetic (non-API) messages.
_SYNTHETIC_MODEL = "<synthetic>"


# ---------------------------------------------------------------------------
# Small parsing helpers.
# ---------------------------------------------------------------------------


def extract_text(content: Any) -> str:
    """Return the textual content of a message body.

    Handles both the plain-string form and the list-of-blocks form, in which
    case only ``text`` blocks are concatenated.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(parts)


def is_human_message(event: dict[str, Any]) -> bool:
    """Return True if the event carries a real human message (non-empty text).

    Both content forms (string and list of blocks) require non-empty text.
    """
    message = event.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "text" and bool(str(b.get("text", "")).strip())
        for b in content
    )


def project_name(cwd: str) -> str:
    """Return the final path component of ``cwd`` (the project folder name).

    Windows paths are recognized by their backslashes so that logs written on
    Windows parse correctly when read from Linux/WSL (and vice versa).
    """
    if not cwd:
        return ""
    if "\\" in cwd:
        return PureWindowsPath(cwd).name
    return PurePosixPath(cwd).name


def prompt_skip_reason(event: dict[str, Any], text: str) -> str | None:
    """Classify a user event that must NOT become a prompt row (3.5/3.6).

    Returns a short reason key, or ``None`` for a genuine human prompt.
    """
    if event.get("isSidechain"):
        return "sidechain"
    if event.get("isMeta"):
        return "meta"
    if event.get("isCompactSummary") or text.startswith(_CONTINUATION_PREFIX):
        return "compact_continuation"
    if any(tag in text for tag in _COMMAND_TAGS):
        return "local_command"
    if text.lstrip().startswith(_INTERRUPT_PREFIX):
        return "interrupted"
    return None


def _usage_tokens(usage: dict[str, Any]) -> dict[str, int]:
    """Map a raw ``message.usage`` payload onto machine token types.

    Cache writes use the per-TTL breakdown (``usage.cache_creation``) when
    present; otherwise the legacy total falls back to ``cache_write_5m``.
    ``server_tool_use`` counts server-side tool *requests*, not tokens.
    Only non-zero entries are returned.
    """

    def _int(value: Any) -> int:
        return int(value) if isinstance(value, (int, float)) else 0

    counts = {
        "input": _int(usage.get("input_tokens")),
        "output": _int(usage.get("output_tokens")),
        "cache_read": _int(usage.get("cache_read_input_tokens")),
    }
    total_write = _int(usage.get("cache_creation_input_tokens"))
    breakdown = usage.get("cache_creation")
    if isinstance(breakdown, dict):
        counts["cache_write_5m"] = _int(breakdown.get("ephemeral_5m_input_tokens"))
        counts["cache_write_1h"] = _int(breakdown.get("ephemeral_1h_input_tokens"))
        if counts["cache_write_5m"] + counts["cache_write_1h"] == 0 and total_write:
            counts["cache_write_5m"] = total_write
    else:
        counts["cache_write_5m"] = total_write
        counts["cache_write_1h"] = 0
    server = usage.get("server_tool_use")
    if isinstance(server, dict):
        counts["server_tool_use"] = sum(
            _int(v) for k, v in server.items() if k.endswith("_requests")
        )
    return {key: value for key, value in counts.items() if value}


def _dedup_key(message: dict[str, Any], event: dict[str, Any]) -> str:
    """Global deduplication key for an assistant usage line.

    ``message.id + requestId`` when both exist (the ccusage rule); degraded
    single-field keys otherwise. Empty string means "never deduplicate".
    """
    message_id = str(message.get("id") or "")
    request_id = str(event.get("requestId") or "")
    if message_id and request_id:
        return f"{message_id}:{request_id}"
    if request_id:
        return f"req:{request_id}"
    if message_id:
        return f"msg:{message_id}"
    return ""


def _parse_ts(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware datetime (UTC if naive)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ---------------------------------------------------------------------------
# Per-file parsing (streaming, one pass, no filtering -- cacheable).
# ---------------------------------------------------------------------------


def parse_file(filepath: Path) -> ParsedFile:
    """Parse one JSONL session file in a single streaming pass.

    Returns the cache-friendly :class:`~prompt_analytics.schema.ParsedFile`
    shape: session metadata, real human prompts, and every assistant usage
    record (with dedup keys -- global dedup happens at aggregation time, as
    duplicates span files). Date filtering is deliberately NOT applied here so
    cached results stay valid for any filter.

    Raises:
        OSError: If the file cannot be opened/read (caller skips the file).
        UnicodeDecodeError: If the file is not valid UTF-8 (idem).
    """
    session_id = ""
    session_cwd = ""
    session_branch = ""
    first_timestamp = ""
    prompts: list[ParsedPrompt] = []
    usage_records: list[UsageRecord] = []
    lines_total = 0
    lines_invalid = 0
    event_types: Counter[str] = Counter()
    filtered_prompts: Counter[str] = Counter()
    versions: set[str] = set()

    # uuid -> prompt id mapping: newer formats only carry ``promptId`` on user
    # events; assistant events are attributed by walking ``parentUuid``.
    uuid_to_pid: dict[str, str] = {}
    # uuid -> "descends from a post-compaction continuation" (1.4): set on the
    # synthetic continuation message, inherited down the chain, and reset by
    # the next real human prompt.
    uuid_post_compact: dict[str, bool] = {}
    seen_prompt_ids: set[str] = set()
    current_mode = "normal"

    with filepath.open(encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            lines_total += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                lines_invalid += 1
                continue
            if not isinstance(event, dict):
                lines_invalid += 1
                continue

            etype = str(event.get("type") or "")
            event_types[etype] += 1

            if not session_id and event.get("sessionId"):
                session_id = str(event["sessionId"])
            if not first_timestamp and event.get("timestamp"):
                first_timestamp = str(event["timestamp"])
            if not session_cwd and event.get("cwd"):
                session_cwd = str(event["cwd"])
                session_branch = str(event.get("gitBranch") or "")
            if event.get("version"):
                versions.add(str(event["version"]))

            # Every event participates in the uuid -> promptId chain (tool
            # results and attachments sit between a prompt and its replies).
            event_uuid = str(event.get("uuid") or "")
            inherited = uuid_to_pid.get(str(event.get("parentUuid") or ""), "")
            pid = str(event.get("promptId") or "") or inherited
            if event_uuid:
                uuid_to_pid[event_uuid] = pid
            post_compact = uuid_post_compact.get(str(event.get("parentUuid") or ""), False)
            if event_uuid:
                uuid_post_compact[event_uuid] = post_compact

            if etype == "mode":
                current_mode = str(event.get("mode") or current_mode)
            elif etype == "user":
                if not is_human_message(event):
                    continue  # tool results and other non-human user events
                text = extract_text(event.get("message", {}).get("content", ""))
                reason = prompt_skip_reason(event, text)
                if reason is not None:
                    filtered_prompts[reason] += 1
                    if reason == "compact_continuation" and event_uuid:
                        uuid_post_compact[event_uuid] = True
                    continue
                # A real human prompt ends any post-compaction continuation.
                if event_uuid:
                    uuid_post_compact[event_uuid] = False
                own_pid = str(event.get("promptId") or "") or event_uuid
                if not own_pid or own_pid in seen_prompt_ids:
                    continue
                seen_prompt_ids.add(own_pid)
                prompts.append(
                    ParsedPrompt(
                        prompt_id=own_pid,
                        timestamp=str(event.get("timestamp") or ""),
                        cwd=str(event.get("cwd") or session_cwd),
                        git_branch=str(event.get("gitBranch") or session_branch),
                        mode=current_mode,
                        entrypoint=str(event.get("entrypoint") or ""),
                        version=str(event.get("version") or ""),
                        text=text,
                    )
                )
            elif etype == "assistant":
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                usage = message.get("usage")
                tokens = _usage_tokens(usage) if isinstance(usage, dict) else {}
                content = message.get("content")
                tool_use_ids = [
                    str(block["id"])
                    for block in (content if isinstance(content, list) else [])
                    if isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("id")
                ]
                key = _dedup_key(message, event)
                if not tokens and not tool_use_ids and not key:
                    continue  # nothing countable (e.g. bare synthetic notices)
                usage_records.append(
                    UsageRecord(
                        prompt_id=pid,
                        dedup_key=key,
                        timestamp=str(event.get("timestamp") or ""),
                        model=str(message.get("model") or ""),
                        stop_reason=str(message.get("stop_reason") or ""),
                        is_sidechain=bool(event.get("isSidechain")),
                        post_compact=post_compact,
                        tokens=tokens,
                        tool_use_ids=tool_use_ids,
                    )
                )

    return ParsedFile(
        session_id=session_id or filepath.stem,
        cwd=session_cwd,
        git_branch=session_branch,
        first_timestamp=first_timestamp,
        prompts=prompts,
        usage=usage_records,
        lines_total=lines_total,
        lines_invalid=lines_invalid,
        event_types=dict(event_types),
        filtered_prompts=dict(filtered_prompts),
        versions=sorted(versions),
    )


# ---------------------------------------------------------------------------
# Parse cache (D2): path + mtime + size -> ParsedFile as JSON.
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    return paths.parse_cache_dir()


def _gc_parse_cache(expected_digests: set[str]) -> None:
    """Drop cache entries whose source JSONL no longer exists (orphan GC).

    The cache key is the SHA-1 of the file path; an entry outside the current
    file set belongs to a deleted/moved log and would otherwise live forever.
    Best-effort like the rest of the cache: failures are ignored.
    """
    with contextlib.suppress(OSError):
        for entry in _cache_dir().glob("*.json"):
            if entry.stem not in expected_digests:
                with contextlib.suppress(OSError):
                    entry.unlink()


def _load_or_parse(filepath: Path, use_cache: bool, report: ExtractReport) -> ParsedFile:
    """Return the parsed file, served from the mtime/size cache when valid."""
    if not use_cache:
        return parse_file(filepath)

    stat = filepath.stat()
    digest = hashlib.sha1(str(filepath).encode("utf-8")).hexdigest()
    cache_path = _cache_dir() / f"{digest}.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("version") == CACHE_VERSION
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and cached.get("size") == stat.st_size
            and isinstance(cached.get("data"), dict)
        ):
            report.files_cached += 1
            return cast("ParsedFile", cached["data"])
    except (OSError, json.JSONDecodeError):
        pass

    data = parse_file(filepath)
    # The cache is best-effort; extraction must not fail on it.
    with contextlib.suppress(OSError):
        atomic_write_json(
            cache_path,
            {
                "version": CACHE_VERSION,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "data": data,
            },
        )
    return data


# ---------------------------------------------------------------------------
# Extraction report (3.16): loud by design.
# ---------------------------------------------------------------------------


@dataclass
class ExtractReport:
    """Everything a user needs to trust (or distrust) an extraction run."""

    files_read: int = 0
    files_cached: int = 0
    files_skipped: list[tuple[str, str]] = field(default_factory=list)
    lines_total: int = 0
    lines_invalid: int = 0
    unknown_event_types: dict[str, int] = field(default_factory=dict)
    sessions: int = 0
    prompts: int = 0
    prompts_total: int = 0
    filtered_prompts: dict[str, int] = field(default_factory=dict)
    usage_records: int = 0
    deduplicated_records: int = 0
    token_rows: int = 0
    request_rows: int = 0
    models: dict[str, int] = field(default_factory=dict)
    unpriced_models: list[str] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)
    window_note: str = ""
    output_dir: str = ""

    @property
    def warnings(self) -> list[str]:
        """Human-readable warnings; non-empty output fails under ``--strict``."""
        messages: list[str] = []
        for path, reason in self.files_skipped:
            messages.append(f"skipped unreadable file {path}: {reason}")
        if self.lines_invalid:
            messages.append(f"{self.lines_invalid} JSONL line(s) were not valid JSON")
        if self.unknown_event_types:
            listed = ", ".join(
                f"{name} (x{count})" for name, count in sorted(self.unknown_event_types.items())
            )
            messages.append(
                f"unknown event type(s) encountered: {listed} -- "
                "the Claude Code format may have evolved"
            )
        if self.unpriced_models:
            messages.append(
                "model(s) without Anthropic pricing: "
                + ", ".join(self.unpriced_models)
                + " -- add an entry or a prefix fallback to pricing.yml"
            )
        if self.files_read and self.prompts_total == 0:
            messages.append(
                f"{self.files_read} file(s) read but 0 prompts extracted -- "
                "the JSONL format may have changed; please report this"
            )
        return messages

    def exit_code(self, *, strict: bool = False) -> int:
        """0 on success; 1 on the zero-prompt canary or any warning in strict mode."""
        if self.files_read and self.prompts_total == 0:
            return 1
        if strict and self.warnings:
            return 1
        return 0

    def format_lines(self) -> list[str]:
        """Render the report for terminal output."""
        lines = ["=== Extraction report ==="]
        cached = f" ({self.files_cached} from cache)" if self.files_cached else ""
        lines.append(f"Files read:      {self.files_read}{cached}")
        lines.append(f"Lines parsed:    {self.lines_total}")
        lines.append(f"Sessions:        {self.sessions}")
        filtered = sum(self.filtered_prompts.values())
        detail = ""
        if filtered:
            parts = ", ".join(
                f"{name}={count}" for name, count in sorted(self.filtered_prompts.items())
            )
            detail = f" (non-prompts filtered: {parts})"
        lines.append(f"Prompts:         {self.prompts}{detail}")
        lines.append(
            f"Usage records:   {self.usage_records}"
            f" ({self.deduplicated_records} duplicate(s) removed)"
        )
        lines.append(f"Token rows:      {self.token_rows}")
        lines.append(f"Request rows:    {self.request_rows}")
        if self.models:
            listed = ", ".join(f"{name} ({count})" for name, count in sorted(self.models.items()))
            lines.append(f"Models:          {listed}")
        if self.versions:
            lines.append(f"Claude Code:     versions {', '.join(self.versions)}")
        if self.window_note:
            lines.append(self.window_note)
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        if self.output_dir:
            lines.append(f"Output:          {self.output_dir}")
        return lines


# ---------------------------------------------------------------------------
# File discovery.
# ---------------------------------------------------------------------------


def _iter_jsonl_files(claude_dir: Path | None = None) -> list[Path]:
    """Return all session JSONL files under the Claude projects directory.

    Defaults to ``~/.claude/projects``, honoring ``CLAUDE_CONFIG_DIR`` (08 m2).
    Subagent transcripts (``**/subagents/*.jsonl``) are INCLUDED: their cost
    belongs to the parent session (policy 3.6).
    """
    base = claude_dir if claude_dir is not None else paths.claude_projects_dir()
    if not base.exists():
        return []
    return sorted(base.rglob("*.jsonl"))


def _subagent_parent_session(filepath: Path) -> str:
    """For ``<...>/<parent-session-id>/subagents/agent-x.jsonl``, the parent id."""
    parts = filepath.parts
    try:
        index = parts.index("subagents")
    except ValueError:
        return ""
    return parts[index - 1] if index >= 1 else ""


# ---------------------------------------------------------------------------
# Aggregation + output.
# ---------------------------------------------------------------------------


def _resolve_tz(timezone_name: str | None) -> tzinfo:
    """Resolve ``--timezone`` (IANA name) or default to the local timezone."""
    if timezone_name:
        try:
            return ZoneInfo(timezone_name)
        except Exception as exc:
            raise ValueError(
                f"Unknown timezone {timezone_name!r}; use an IANA name like 'Europe/Paris'."
            ) from exc
    local = datetime.now().astimezone().tzinfo
    return local if local is not None else timezone.utc


def _parse_bound(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid --{label} date {value!r}; expected YYYY-MM-DD.") from exc


@dataclass
class _PromptAgg:
    """Aggregated state for one prompt across all files."""

    session_id: str
    parsed: ParsedPrompt
    # (model, token_type, is_sidechain) -> count: pricing is per model, and a
    # prompt can mix models (e.g. Haiku subcalls under an Opus session);
    # sidechain usage is kept apart as its own dimension (1.2).
    tokens: Counter[tuple[str, str, int]] = field(default_factory=Counter)
    records: list[UsageRecord] = field(default_factory=list)  # non-sidechain
    tool_call_count: int = 0


def _request_rows(session_id: str, prompt_id: str, records: list[UsageRecord]) -> list[RequestRow]:
    """Pivot a prompt's deduplicated usage records into requests.csv rows.

    Chronological order within the prompt; per prompt, the column sums equal
    the tokens.csv totals exactly (V7) because both come from the same records.
    """
    rows: list[RequestRow] = []
    for index, record in enumerate(sorted(records, key=lambda r: r["timestamp"]), start=1):
        tokens = record["tokens"]
        rows.append(
            RequestRow(
                session_id=session_id,
                prompt_id=prompt_id,
                request_index=index,
                timestamp=record["timestamp"],
                model=record["model"],
                stop_reason=record["stop_reason"],
                is_sidechain=1 if record["is_sidechain"] else 0,
                post_compact=1 if record["post_compact"] else 0,
                input_tokens=tokens.get("input", 0),
                output_tokens=tokens.get("output", 0),
                cache_read_tokens=tokens.get("cache_read", 0),
                cache_write_5m_tokens=tokens.get("cache_write_5m", 0),
                cache_write_1h_tokens=tokens.get("cache_write_1h", 0),
                server_tool_use_requests=tokens.get("server_tool_use", 0),
            )
        )
    return rows


@dataclass
class ExtractResult:
    """The full in-memory result of one extraction pass.

    Consumed by :func:`run_extract` (which writes the CSVs) and by the
    analytics layer's on-the-fly mode (which never touches the disk).
    """

    report: ExtractReport
    sessions: list[SessionRow]
    prompts: list[PromptRow]
    tokens: list[TokenRow]
    requests: list[RequestRow]
    texts: list[dict[str, str]]


def collect(
    *,
    no_text: bool = False,
    since: str | None = None,
    until: str | None = None,
    timezone_name: str | None = None,
    use_cache: bool = True,
    claude_dir: Path | None = None,
) -> ExtractResult:
    """Parse and aggregate the full local history into in-memory rows.

    This is :func:`run_extract` minus the CSV writes; ``since``/``until`` only
    restrict the returned rows.

    Args:
        no_text: Skip collecting full prompt texts AND blank out the
            ``prompt_preview`` column (honest privacy switch, 10.1).
        since: Lower date bound (``YYYY-MM-DD``), inclusive, at prompt grain.
        until: Upper date bound (``YYYY-MM-DD``), inclusive up to 23:59:59.999999.
        timezone_name: IANA timezone for interpreting the date bounds and
            session dates (default: the local timezone).
        use_cache: Serve unchanged files from the parse cache.
        claude_dir: Override of ``~/.claude/projects`` (mainly for tests).

    Returns:
        An :class:`ExtractResult` with the report and all output rows.

    Raises:
        ValueError: On an invalid date bound or timezone name.
    """
    tz = _resolve_tz(timezone_name)
    since_dt = (
        datetime.combine(_parse_bound(since, "since"), datetime.min.time(), tz) if since else None
    )
    until_dt = (
        datetime.combine(_parse_bound(until, "until"), datetime.max.time(), tz) if until else None
    )

    report = ExtractReport()

    # --- Pass over all files: parse (or load from cache) and merge. ---------
    sessions: dict[str, SessionRow] = {}
    session_first_ts: dict[str, str] = {}
    prompt_aggs: dict[str, _PromptAgg] = {}
    prompt_order: list[str] = []
    # (sid, pid) -> (model, token_type, is_sidechain) -> count
    overhead_tokens: dict[tuple[str, str], Counter[tuple[str, str, int]]] = {}
    overhead_in_window: set[tuple[str, str]] = set()
    # (sid, output prompt id) -> the deduplicated records behind it (1.1).
    request_records: dict[tuple[str, str], list[UsageRecord]] = defaultdict(list)
    seen_tool_ids: set[str] = set()
    unknown_types: Counter[str] = Counter()
    filtered_prompts: Counter[str] = Counter()
    models: Counter[str] = Counter()
    versions: set[str] = set()
    pending_usage: list[tuple[str, UsageRecord]] = []  # (session_id, record)

    def _in_window(ts: datetime | None) -> bool:
        if ts is None:
            return not (since_dt or until_dt)
        if since_dt and ts < since_dt:
            return False
        return not (until_dt and ts > until_dt)

    jsonl_files = _iter_jsonl_files(claude_dir)
    for filepath in jsonl_files:
        try:
            parsed = _load_or_parse(filepath, use_cache, report)
        except (OSError, UnicodeDecodeError) as exc:
            report.files_skipped.append((str(filepath), f"{type(exc).__name__}: {exc}"))
            continue
        report.files_read += 1
        report.lines_total += parsed["lines_total"]
        report.lines_invalid += parsed["lines_invalid"]
        for etype, count in parsed["event_types"].items():
            if etype not in KNOWN_EVENT_TYPES:
                unknown_types[etype] += count
        for reason, count in parsed["filtered_prompts"].items():
            filtered_prompts[reason] += count
        versions.update(parsed["versions"])

        session_id = _subagent_parent_session(filepath) or parsed["session_id"]
        if session_id not in sessions and parsed["cwd"]:
            sessions[session_id] = SessionRow(
                session_id=session_id,
                start_date="",
                project=project_name(parsed["cwd"]),
                cwd=parsed["cwd"],
                git_branch=parsed["git_branch"],
            )
        first_ts = parsed["first_timestamp"]
        if first_ts and (
            session_id not in session_first_ts or first_ts < session_first_ts[session_id]
        ):
            session_first_ts[session_id] = first_ts

        # Prompts: first occurrence wins globally (a resumed session replays
        # the same promptIds into a new file).
        for prompt in parsed["prompts"]:
            pid = prompt["prompt_id"]
            if pid in prompt_aggs:
                continue
            prompt_aggs[pid] = _PromptAgg(session_id=session_id, parsed=prompt)
            prompt_order.append(pid)

        for record in parsed["usage"]:
            pending_usage.append((session_id, record))

    if use_cache:
        digests = {
            hashlib.sha1(str(filepath).encode("utf-8")).hexdigest() for filepath in jsonl_files
        }
        _gc_parse_cache(digests)

    # --- Global dedup (after all prompts are known). -------------------------
    # A message is written as several JSONL lines (one content block each),
    # every line repeating a *progressive* usage snapshot -- output_tokens
    # grows line after line, and even the model can change mid-message. Per
    # message.id+requestId key (globally across files: resumed sessions replay
    # identical copies) we keep the LARGEST snapshot; on ties, the first line
    # (a message can straddle midnight -- the request belongs to the day it
    # started). This exactly matches ccusage, as validated empirically by
    # scripts/reconcile_ccusage.py. Tool calls are the union over all lines
    # (each line carries one block), deduplicated by tool_use id.
    def _magnitude(record: UsageRecord) -> int:
        return sum(record["tokens"].values())

    chosen: dict[str, tuple[str, UsageRecord]] = {}
    keyless: list[tuple[str, UsageRecord]] = []
    tool_ids_by_prompt: dict[str, list[str]] = defaultdict(list)
    for session_id, record in pending_usage:
        if not record["is_sidechain"] and record["prompt_id"] in prompt_aggs:
            for tool_id in record["tool_use_ids"]:
                if tool_id not in seen_tool_ids:
                    seen_tool_ids.add(tool_id)
                    tool_ids_by_prompt[record["prompt_id"]].append(tool_id)
        key = record["dedup_key"]
        if not key:
            keyless.append((session_id, record))
            continue
        previous = chosen.get(key)
        if previous is None:
            chosen[key] = (session_id, record)
        else:
            report.deduplicated_records += 1
            if _magnitude(record) > _magnitude(previous[1]):
                chosen[key] = (session_id, record)

    # --- Attribution. ---------------------------------------------------------
    for session_id, record in [*chosen.values(), *keyless]:
        report.usage_records += 1
        if record["model"]:
            models[record["model"]] += 1

        pid = record["prompt_id"]
        agg = prompt_aggs.get(pid)
        side = 1 if record["is_sidechain"] else 0
        if agg is not None:
            # Sidechain usage is attached to the parent prompt's cost but
            # excluded from assistant_turns / tool_calls (3.6).
            for token_type, count in record["tokens"].items():
                agg.tokens[(record["model"], token_type, side)] += count
            if not record["is_sidechain"]:
                agg.records.append(record)
            if record["tokens"]:
                request_records[(agg.session_id, pid)].append(record)
        else:
            # Session overhead: continuation tails (no prompt id) keep the
            # ``:_continuation`` pseudo id; filtered fake prompts keep theirs.
            pseudo = pid or continuation_prompt_id(session_id)
            bucket = overhead_tokens.setdefault((session_id, pseudo), Counter())
            for token_type, count in record["tokens"].items():
                bucket[(record["model"], token_type, side)] += count
            if record["tokens"]:
                request_records[(session_id, pseudo)].append(record)
            if _in_window(_parse_ts(record["timestamp"])):
                overhead_in_window.add((session_id, pseudo))

    for pid, tool_ids in tool_ids_by_prompt.items():
        prompt_aggs[pid].tool_call_count = len(tool_ids)

    # --- Build rows (date filter applies at prompt grain, 3.9). -------------
    prompts_by_session: dict[str, list[str]] = defaultdict(list)
    for pid in prompt_order:
        prompts_by_session[prompt_aggs[pid].session_id].append(pid)

    prompt_rows: list[PromptRow] = []
    token_rows: list[TokenRow] = []
    request_rows: list[RequestRow] = []
    text_rows: list[dict[str, str]] = []
    included_sessions: set[str] = set()

    for session_id, pids in prompts_by_session.items():
        pids.sort(key=lambda p: prompt_aggs[p].parsed["timestamp"])
        # prompt_index reflects the position in the full session, independent
        # of the date window.
        for index, pid in enumerate(pids, start=1):
            agg = prompt_aggs[pid]
            prompt = agg.parsed
            if not _in_window(_parse_ts(prompt["timestamp"])):
                continue
            included_sessions.add(session_id)
            agg.records.sort(key=lambda r: r["timestamp"])
            model = next(
                (r["model"] for r in agg.records if r["model"] and r["model"] != _SYNTHETIC_MODEL),
                "",
            )
            stop_reason = next(
                (r["stop_reason"] for r in reversed(agg.records) if r["stop_reason"]),
                "",
            )
            # --no-text is honest about privacy: it suppresses the preview too,
            # not just prompts_text.csv (10.1).
            preview = (
                ""
                if no_text
                else prompt["text"][:PREVIEW_CHARS].replace("\n", " ").replace("\r", " ")
            )
            prompt_rows.append(
                PromptRow(
                    session_id=session_id,
                    prompt_id=pid,
                    prompt_index=index,
                    timestamp=prompt["timestamp"],
                    project=project_name(prompt["cwd"]),
                    cwd=prompt["cwd"],
                    git_branch=prompt["git_branch"],
                    mode=prompt["mode"],
                    entrypoint=prompt["entrypoint"],
                    version=prompt["version"],
                    model=model,
                    char_count=len(prompt["text"]),
                    assistant_turns=len(agg.records),
                    tool_calls=agg.tool_call_count,
                    final_stop_reason=stop_reason,
                    prompt_preview=preview,
                )
            )
            for (token_model, token_type, side), count in agg.tokens.items():
                if count:
                    token_rows.append(
                        TokenRow(
                            session_id=session_id,
                            prompt_id=pid,
                            model=token_model,
                            token_type=token_type,
                            is_sidechain=side,
                            token_count=count,
                        )
                    )
            request_rows.extend(
                _request_rows(session_id, pid, request_records.get((session_id, pid), []))
            )
            if not no_text:
                text_rows.append({"prompt_id": pid, "prompt_text": prompt["text"]})

    for (session_id, pseudo), counts in overhead_tokens.items():
        if (session_id, pseudo) not in overhead_in_window:
            continue
        included_sessions.add(session_id)
        for (token_model, token_type, side), count in counts.items():
            if count:
                token_rows.append(
                    TokenRow(
                        session_id=session_id,
                        prompt_id=pseudo,
                        model=token_model,
                        token_type=token_type,
                        is_sidechain=side,
                        token_count=count,
                    )
                )
        request_rows.extend(
            _request_rows(session_id, pseudo, request_records.get((session_id, pseudo), []))
        )

    session_rows: list[SessionRow] = []
    for session_id in included_sessions:
        row = sessions.get(session_id)
        if row is None:
            row = SessionRow(
                session_id=session_id, start_date="", project="", cwd="", git_branch=""
            )
        start = _parse_ts(session_first_ts.get(session_id, ""))
        row["start_date"] = start.astimezone(tz).date().isoformat() if start else ""
        session_rows.append(row)

    session_rows.sort(key=lambda r: (r["start_date"], r["session_id"]))
    prompt_rows.sort(key=lambda r: (r["timestamp"], r["prompt_id"]))
    token_order = {name: i for i, name in enumerate(TOKEN_TYPES)}
    token_rows.sort(
        key=lambda r: (r["prompt_id"], r["model"], token_order[r["token_type"]], r["is_sidechain"])
    )
    # Session-chronological: the natural order for gap/TTL analyses.
    request_rows.sort(
        key=lambda r: (r["session_id"], r["timestamp"], r["prompt_id"], r["request_index"])
    )
    text_rows.sort(key=lambda r: r["prompt_id"])

    # --- Finalize the report. ------------------------------------------------
    report.sessions = len(session_rows)
    report.prompts = len(prompt_rows)
    report.prompts_total = len(prompt_aggs)
    report.token_rows = len(token_rows)
    report.request_rows = len(request_rows)
    report.unknown_event_types = dict(unknown_types)
    report.filtered_prompts = dict(filtered_prompts)
    report.models = dict(models)
    report.versions = sorted(versions)
    report.unpriced_models = sorted(
        model
        for model in models
        if model != _SYNTHETIC_MODEL and get_model_pricing(model, "anthropic") is None
    )
    if since_dt or until_dt:
        report.window_note = (
            f"Date filter:     kept {report.prompts} of {report.prompts_total} prompts "
            f"({since or '...'} .. {until or '...'}, {tz})"
        )
    return ExtractResult(
        report=report,
        sessions=session_rows,
        prompts=prompt_rows,
        tokens=token_rows,
        requests=request_rows,
        texts=text_rows,
    )


def run_extract(
    output_dir: Path,
    *,
    no_text: bool = False,
    since: str | None = None,
    until: str | None = None,
    timezone_name: str | None = None,
    use_cache: bool = True,
    claude_dir: Path | None = None,
) -> ExtractReport:
    """Extract usage records from local JSONL files into CSV outputs.

    Always regenerates every CSV atomically from the full history (D2);
    ``since``/``until`` only restrict the *output*, never a persistent store.

    Args:
        output_dir: Directory where CSV outputs are written (created if missing).
        no_text: Skip writing ``prompts_text.csv`` (a stale one is removed) and
            blank out the ``prompt_preview`` column in ``prompts.csv`` (10.1).
        since: Lower date bound (``YYYY-MM-DD``), inclusive, at prompt grain.
        until: Upper date bound (``YYYY-MM-DD``), inclusive up to 23:59:59.999999.
        timezone_name: IANA timezone for interpreting the date bounds and
            session dates (default: the local timezone).
        use_cache: Serve unchanged files from the parse cache.
        claude_dir: Override of ``~/.claude/projects`` (mainly for tests).

    Returns:
        The :class:`ExtractReport`; callers decide the process exit code via
        :meth:`ExtractReport.exit_code`.

    Raises:
        ValueError: On an invalid date bound or timezone name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = collect(
        no_text=no_text,
        since=since,
        until=until,
        timezone_name=timezone_name,
        use_cache=use_cache,
        claude_dir=claude_dir,
    )

    # --- Atomic writes (3.14). ----------------------------------------------
    atomic_write_csv(output_dir / "sessions.csv", SESSIONS_COLS, result.sessions)
    atomic_write_csv(output_dir / "prompts.csv", PROMPTS_COLS, result.prompts)
    atomic_write_csv(output_dir / "tokens.csv", TOKENS_COLS, result.tokens)
    atomic_write_csv(output_dir / "requests.csv", REQUESTS_COLS, result.requests)
    # Window marker (1.5): a windowed extract must never be served later as if
    # it covered the full history -- readers append it to their Source line.
    atomic_write_json(
        output_dir / "extract_meta.json",
        {
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "window": (
                {"since": since, "until": until, "timezone": timezone_name}
                if since or until
                else None
            ),
        },
    )
    atomic_write_csv(
        output_dir / "token_types.csv",
        TOKEN_TYPES_COLS,
        [
            {
                "token_type": token_type,
                "label": TOKEN_TYPE_LABELS[token_type],
                "description": TOKEN_TYPE_DESCRIPTIONS[token_type],
            }
            for token_type in TOKEN_TYPES
        ],
    )
    text_path = output_dir / "prompts_text.csv"
    if no_text:
        text_path.unlink(missing_ok=True)  # never leave stale text behind
    else:
        atomic_write_csv(text_path, PROMPT_TEXT_COLS, result.texts)

    result.report.output_dir = str(output_dir)
    return result.report
