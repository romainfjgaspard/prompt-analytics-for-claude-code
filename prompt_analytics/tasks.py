"""Task attribution (Axe B2): the unit of work is the task, not the prompt.

The parlant level of aggregation is neither the prompt nor the category but the
**task** -- "implement feature X", "debug Y". The prompt that *launches* the work
is the centre of gravity; the ones around it (prepare, plan, then refine, "add
the tests", "commit") gravitate to the same task. This module assembles those
tasks and maps every prompt to one, so the cost machinery can answer "this task
cost you $X, of which $Y is context rent" -- the niveau no tool gives with cost
rigour (cc-lens has a *todo browser* with no cost attribution; our moat is the
cost attached to the task plus category awareness).

Two assembly paths, decided 2026-06-20 (``PLAN-categorisation-et-composition``
§5quinquies), in honest order of trust:

* **Spine = the real ``TodoWrite`` todos.** When a session has todos, the task is
  *recorded*, not guessed: each distinct ``in_progress`` label is a task and each
  prompt joins whichever todo was active during its turn. Faithful to the
  "measure the real" ADN of Axe C (real files) and D (real context).
* **Fallback = inference.** When a session has no todos, a task is a coherent run
  of prompts: a time gap opens a new one, and -- when an embedder is supplied --
  a fresh *anchoring* prompt (plan/implementation) that is semantically far from
  the running segment also opens one. The gap rule alone is fully offline and
  deterministic; the embedding rule sharpens the boundaries (it reuses the B1
  socle). Category structure (anchors vs trailers) decides which prompt *names*
  the task, never splitting on a trailer ("ok go", "commit").

**Caveat, kept honest like the rest:** a todo is *Claude's* decomposition, not
always the user's mental task; the inference is an estimate (fuzzy boundaries).
``origin`` records which path produced each task so readers can weigh it.

The cost of a task is **not** computed here: every real prompt lands in exactly
one task, so a task's input+output+cache cost is just its prompts' token rows
priced at read time -- it reconciles to the bill by construction (see
``analytics``). This module only decides *membership* and *identity*.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .schema import TaskPromptRow, TaskRow

if TYPE_CHECKING:
    from .embeddings import Embedder

__all__ = [
    "TaskPromptInput",
    "TaskTodoInput",
    "ANCHOR_CATEGORIES",
    "DEFAULT_GAP_SECONDS",
    "DEFAULT_SIM_THRESHOLD",
    "assemble_tasks",
]

# Categories that *start* a task (a new piece of work begins). Everything else
# (debug/refactor/review/test/docs/ops/question/followup/feedback/...) gravitates
# to the task in progress -- a trailer never opens a task on its own. Used by the
# fallback to pick the prompt that names a segment and, with an embedder, to
# decide a semantic boundary.
ANCHOR_CATEGORIES = frozenset({"plan", "implementation"})

# A silence longer than this (seconds) between two prompts of a session opens a
# new task in the fallback path -- the strongest workload-agnostic signal that a
# new piece of work began. 30 minutes by default.
DEFAULT_GAP_SECONDS = 1800

# Cosine below this between an anchoring prompt and the running segment's
# centroid marks a semantically distinct new task (embedder path only). The
# embedding ceiling on intent is ~0.7, so this stays deliberately permissive:
# only a clearly unrelated anchor splits without a time gap.
DEFAULT_SIM_THRESHOLD = 0.35

# Task names are short, single-line snippets (todo labels are already short).
_NAME_MAX_CHARS = 80
_WS_RE = re.compile(r"\s+")


class TaskPromptInput(NamedTuple):
    """One real prompt handed to the assembler (the only fields it needs).

    ``text`` is the full prompt text (used by the fallback for the heuristic
    category, the embedding and the name); it is never persisted by this module.
    """

    session_id: str
    prompt_id: str
    timestamp: str
    text: str


class TaskTodoInput(NamedTuple):
    """One deduplicated ``TodoWrite`` snapshot, tagged with its session.

    The active todo ``label`` at ``timestamp`` (empty when nothing is
    ``in_progress``). Deduplication and session resolution happen upstream (in
    ``extract.collect``); the assembler only reads these three fields.
    """

    session_id: str
    timestamp: str
    label: str


class _Segment(NamedTuple):
    """An assembled task before it gets an id: its name + member prompts."""

    name: str
    prompts: list[TaskPromptInput]


def _parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware datetime (``datetime.min`` on
    failure, so ordering/window math never raises)."""
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def _clean_name(text: str) -> str:
    """A short, single-line task name from a label or prompt text."""
    return _WS_RE.sub(" ", text).strip()[:_NAME_MAX_CHARS]


def assemble_tasks(
    prompts: list[TaskPromptInput],
    todos: list[TaskTodoInput],
    *,
    no_text: bool = False,
    embedder: Embedder | None = None,
    gap_seconds: int = DEFAULT_GAP_SECONDS,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
) -> tuple[list[TaskRow], list[TaskPromptRow]]:
    """Assemble tasks and map every prompt to exactly one.

    Per session: if it carries any ``TodoWrite`` with an ``in_progress`` label,
    the **todo spine** owns it (each prompt joins the todo active during its
    turn); otherwise the **fallback** segments it by time gaps (+ semantics when
    ``embedder`` is given). Every prompt of ``prompts`` ends up in one task, so
    the costs reconcile downstream.

    Args:
        prompts: The real prompts to attribute (any order; grouped/sorted here).
        todos: Deduplicated, session-tagged ``TodoWrite`` snapshots (any order).
        no_text: Blank ``inferred`` task names (they derive from user text), like
            the prompt preview under ``--no-text``. Todo labels are kept (they
            are Claude-authored).
        embedder: Optional B1 embedder; enables the semantic split in the
            fallback. ``None`` keeps the fallback gap-only (offline, deterministic).
        gap_seconds: Silence (s) that opens a new fallback task.
        sim_threshold: Cosine below which a fresh anchor opens a new task.

    Returns:
        ``(task_rows, link_rows)`` -- ``tasks.csv`` and ``task_prompts.csv`` rows,
        sorted by ``(session_id, first_timestamp, task_id)`` and
        ``(task_id, prompt_id)`` respectively.
    """
    prompts_by_session: dict[str, list[TaskPromptInput]] = defaultdict(list)
    for prompt in prompts:
        prompts_by_session[prompt.session_id].append(prompt)
    todos_by_session: dict[str, list[TaskTodoInput]] = defaultdict(list)
    for todo in todos:
        todos_by_session[todo.session_id].append(todo)

    task_rows: list[TaskRow] = []
    link_rows: list[TaskPromptRow] = []
    for session_id, session_prompts in prompts_by_session.items():
        ordered = sorted(session_prompts, key=lambda p: (p.timestamp, p.prompt_id))
        session_todos = sorted(todos_by_session.get(session_id, []), key=lambda t: t.timestamp)
        if any(todo.label for todo in session_todos):
            segments = _segment_by_todos(ordered, session_todos)
            origin = "todo"
        else:
            segments = _segment_by_inference(
                ordered,
                embedder=embedder,
                gap_seconds=gap_seconds,
                sim_threshold=sim_threshold,
            )
            origin = "inferred"
        _emit(task_rows, link_rows, session_id, origin, segments, no_text=no_text)

    task_rows.sort(key=lambda r: (r["session_id"], r["first_timestamp"], r["task_id"]))
    link_rows.sort(key=lambda r: (r["task_id"], r["prompt_id"]))
    return task_rows, link_rows


def _segment_by_todos(prompts: list[TaskPromptInput], todos: list[TaskTodoInput]) -> list[_Segment]:
    """Spine: attach each prompt to the todo ``in_progress`` during its turn.

    A prompt's window is ``[its timestamp, the next prompt's timestamp)``; its
    active todo is the last non-empty label written in that window (the turn may
    finish a todo and start the next -- the one left ``in_progress`` wins). Empty
    windows inherit the previous active label (forward fill); a leading prompt
    before the first todo inherits the first label (back fill). One task per
    distinct label, ordered by first appearance, so every prompt gets a task.
    """
    n = len(prompts)
    p_ts = [_parse_ts(p.timestamp) for p in prompts]
    t_ts = [_parse_ts(t.timestamp) for t in todos]

    active: list[str | None] = [None] * n
    for i in range(n):
        lo = p_ts[i]
        hi = p_ts[i + 1] if i + 1 < n else None
        window_label: str | None = None
        for t, todo in zip(t_ts, todos, strict=True):
            if todo.label and t >= lo and (hi is None or t < hi):
                window_label = todo.label  # last in window wins
        active[i] = window_label

    last: str | None = None
    for i in range(n):
        if active[i] is None:
            active[i] = last
        else:
            last = active[i]
    first_known = next((label for label in active if label is not None), "")
    active = [label if label is not None else first_known for label in active]

    order: list[str] = []
    members: dict[str, list[TaskPromptInput]] = defaultdict(list)
    for label, prompt in zip(active, prompts, strict=True):
        key = label or ""
        if key not in members:
            order.append(key)
        members[key].append(prompt)
    return [_Segment(name=label, prompts=members[label]) for label in order]


def _segment_by_inference(
    prompts: list[TaskPromptInput],
    *,
    embedder: Embedder | None,
    gap_seconds: int,
    sim_threshold: float,
) -> list[_Segment]:
    """Fallback: segment a no-todo session into coherent runs of prompts.

    A new task opens on a time gap larger than ``gap_seconds`` or -- with an
    embedder -- on a fresh anchoring prompt (plan/implementation) whose cosine to
    the running segment's centroid is below ``sim_threshold``. Each segment is
    named after its first anchoring prompt (or its first prompt). Deterministic
    when ``embedder`` is ``None`` or a deterministic embedder is used.
    """
    if not prompts:
        return []
    # Heuristic category per prompt (pure regex, offline) for the anchor rule.
    from .categorize import _classify_heuristic

    categories = [_classify_heuristic(p.text) for p in prompts]
    p_ts = [_parse_ts(p.timestamp) for p in prompts]
    vectors = embedder.embed([p.text for p in prompts]) if embedder is not None else None

    segments_idx: list[list[int]] = []
    current: list[int] = [0]
    centroid_sum = vectors[0].copy() if vectors is not None else None
    for i in range(1, len(prompts)):
        gap = (p_ts[i] - p_ts[i - 1]).total_seconds()
        new_task = gap > gap_seconds
        if not new_task and vectors is not None and categories[i] in ANCHOR_CATEGORIES:
            assert centroid_sum is not None
            norm = float(np.linalg.norm(centroid_sum))
            if norm > 0.0:
                similarity = float(vectors[i] @ (centroid_sum / norm))
                if similarity < sim_threshold:
                    new_task = True
        if new_task:
            segments_idx.append(current)
            current = [i]
            centroid_sum = vectors[i].copy() if vectors is not None else None
        else:
            current.append(i)
            if vectors is not None:
                assert centroid_sum is not None
                centroid_sum += vectors[i]
    segments_idx.append(current)

    segments: list[_Segment] = []
    for indices in segments_idx:
        anchor = next((j for j in indices if categories[j] in ANCHOR_CATEGORIES), indices[0])
        segments.append(
            _Segment(
                name=_clean_name(prompts[anchor].text),
                prompts=[prompts[j] for j in indices],
            )
        )
    return segments


def _emit(
    task_rows: list[TaskRow],
    link_rows: list[TaskPromptRow],
    session_id: str,
    origin: str,
    segments: list[_Segment],
    *,
    no_text: bool,
) -> None:
    """Turn assembled segments into ``tasks.csv`` + ``task_prompts.csv`` rows."""
    prefix = "t" if origin == "todo" else "i"
    for index, segment in enumerate(segments, start=1):
        task_id = f"{session_id}:{prefix}{index:02d}"
        # Inferred names come from user text -> blanked under --no-text, like the
        # prompt preview. Todo labels are Claude-authored and always kept.
        name = "" if (origin == "inferred" and no_text) else segment.name
        timestamps = sorted(p.timestamp for p in segment.prompts)
        task_rows.append(
            TaskRow(
                task_id=task_id,
                session_id=session_id,
                name=name,
                origin=origin,
                prompts=len(segment.prompts),
                first_timestamp=timestamps[0],
                last_timestamp=timestamps[-1],
            )
        )
        for prompt in segment.prompts:
            link_rows.append(TaskPromptRow(task_id=task_id, prompt_id=prompt.prompt_id))
