"""Tests for task attribution (Axe B2): the todo spine, the inference fallback,
and the cost-reconciliation invariant (every prompt in exactly one task)."""

from __future__ import annotations

import csv
from pathlib import Path

from prompt_analytics.analytics import CostEngine, dataset_from_csvs
from prompt_analytics.embeddings import HashingEmbedder
from prompt_analytics.extract import run_extract
from prompt_analytics.tasks import (
    TaskPromptInput,
    TaskTodoInput,
    assemble_tasks,
)


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _p(pid, ts, text="do something", sid="S"):
    return TaskPromptInput(session_id=sid, prompt_id=pid, timestamp=ts, text=text)


# ---------------------------------------------------------------------------
# Todo spine: prompts attach to the todo in_progress during their turn.
# ---------------------------------------------------------------------------


def test_todo_spine_attaches_prompts_to_the_active_todo():
    prompts = [
        _p("p1", "2026-06-01T10:00:00.000Z"),
        _p("p2", "2026-06-01T10:10:00.000Z"),
        _p("p3", "2026-06-01T10:20:00.000Z"),
        _p("p4", "2026-06-01T10:30:00.000Z"),
    ]
    todos = [
        TaskTodoInput("S", "2026-06-01T10:01:00.000Z", "Build the feature"),
        TaskTodoInput("S", "2026-06-01T10:22:00.000Z", "Write the tests"),
    ]
    tasks, links = assemble_tasks(prompts, todos)

    assert [t["origin"] for t in tasks] == ["todo", "todo"]
    assert [t["name"] for t in tasks] == ["Build the feature", "Write the tests"]
    by_task = {t["task_id"]: t for t in tasks}
    members = {
        t_id: sorted(e["prompt_id"] for e in links if e["task_id"] == t_id) for t_id in by_task
    }
    # p1/p2 ran while "Build the feature" was active; p3 (which started the next
    # todo) and p4 belong to "Write the tests".
    assert members["S:t01"] == ["p1", "p2"]
    assert members["S:t02"] == ["p3", "p4"]
    assert by_task["S:t01"]["prompts"] == 2
    assert by_task["S:t01"]["first_timestamp"] == "2026-06-01T10:00:00.000Z"
    assert by_task["S:t02"]["last_timestamp"] == "2026-06-01T10:30:00.000Z"


def test_todo_spine_back_fills_a_leading_prompt_before_the_first_todo():
    # The opener runs before any TodoWrite, then a far-later prompt opens a todo;
    # the opener back-fills onto the first known task (never left orphaned).
    prompts = [
        _p("p1", "2026-06-01T09:00:00.000Z"),
        _p("p2", "2026-06-01T09:05:00.000Z"),
    ]
    todos = [TaskTodoInput("S", "2026-06-01T09:06:00.000Z", "Ship it")]
    tasks, links = assemble_tasks(prompts, todos)
    assert len(tasks) == 1
    assert tasks[0]["name"] == "Ship it"
    assert {e["prompt_id"] for e in links} == {"p1", "p2"}


# ---------------------------------------------------------------------------
# Inference fallback: gaps, then (with an embedder) semantics.
# ---------------------------------------------------------------------------


def test_fallback_splits_on_a_time_gap():
    prompts = [
        _p("p1", "2026-06-01T10:00:00.000Z", "implement the export endpoint"),
        _p("p2", "2026-06-01T10:05:00.000Z", "ok continue"),
        _p("p3", "2026-06-01T12:00:00.000Z", "now debug the failing job"),  # ~2h gap
        _p("p4", "2026-06-01T12:10:00.000Z", "fix it"),
    ]
    tasks, links = assemble_tasks(prompts, [])
    assert [t["origin"] for t in tasks] == ["inferred", "inferred"]
    members = {
        t["task_id"]: sorted(e["prompt_id"] for e in links if e["task_id"] == t["task_id"])
        for t in tasks
    }
    assert members["S:i01"] == ["p1", "p2"]
    assert members["S:i02"] == ["p3", "p4"]


def test_fallback_does_not_split_without_a_gap_or_embedder():
    prompts = [
        _p("p1", "2026-06-01T10:00:00.000Z", "plan the export pipeline architecture"),
        _p("p2", "2026-06-01T10:02:00.000Z", "ok go ahead"),
        _p("p3", "2026-06-01T10:05:00.000Z", "implement the login screen module"),
    ]
    tasks, _ = assemble_tasks(prompts, [])
    assert len(tasks) == 1  # one continuous run, gap-only fallback


def test_fallback_embedder_splits_on_a_dissimilar_anchor():
    prompts = [
        _p("p1", "2026-06-01T10:00:00.000Z", "plan the export pipeline architecture"),
        _p("p2", "2026-06-01T10:02:00.000Z", "ok go ahead"),
        _p("p3", "2026-06-01T10:05:00.000Z", "implement the login screen module"),
    ]
    # A high threshold makes any non-near-identical anchor a fresh task; the
    # trailer ("ok go ahead") never triggers the semantic check.
    tasks, links = assemble_tasks(prompts, [], embedder=HashingEmbedder(dim=64), sim_threshold=0.9)
    assert len(tasks) == 2
    members = {
        t["task_id"]: sorted(e["prompt_id"] for e in links if e["task_id"] == t["task_id"])
        for t in tasks
    }
    assert members["S:i01"] == ["p1", "p2"]
    assert members["S:i02"] == ["p3"]


def test_inferred_names_are_blanked_under_no_text():
    prompts = [_p("p1", "2026-06-01T10:00:00.000Z", "implement the secret feature")]
    tasks, _ = assemble_tasks(prompts, [], no_text=True)
    assert tasks[0]["name"] == ""  # derived from user text -> suppressed
    # But a todo label is Claude-authored and kept even under --no-text.
    tasks2, _ = assemble_tasks(
        prompts, [TaskTodoInput("S", "2026-06-01T10:00:00.000Z", "Keep me")], no_text=True
    )
    assert tasks2[0]["name"] == "Keep me"


def test_every_prompt_lands_in_exactly_one_task():
    prompts = [_p(f"p{i}", f"2026-06-01T10:{i:02d}:00.000Z") for i in range(10)]
    _, links = assemble_tasks(prompts, [])
    covered = [e["prompt_id"] for e in links]
    assert sorted(covered) == sorted(p.prompt_id for p in prompts)
    assert len(covered) == len(set(covered))  # no prompt in two tasks


# ---------------------------------------------------------------------------
# Extraction integration + cost reconciliation.
# ---------------------------------------------------------------------------


def _user(pid, ts, text):
    return {
        "type": "user",
        "promptId": pid,
        "uuid": f"u-{pid}",
        "parentUuid": None,
        "timestamp": ts,
        "cwd": "/home/dev/proj",
        "gitBranch": "main",
        "entrypoint": "cli",
        "version": "2.2.0",
        "sessionId": "sess-t",
        "message": {"role": "user", "content": text},
    }


def _assistant(req, ts, parent, content, inp=100, out=50):
    return {
        "type": "assistant",
        "uuid": f"u-{req}",
        "parentUuid": parent,
        "requestId": req,
        "timestamp": ts,
        "sessionId": "sess-t",
        "cwd": "/home/dev/proj",
        "message": {
            "id": f"m-{req}",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": content,
            "usage": {"input_tokens": inp, "output_tokens": out},
        },
    }


def _todowrite(label):
    return {
        "type": "tool_use",
        "id": f"tw-{label[:4]}",
        "name": "TodoWrite",
        "input": {"todos": [{"content": label, "status": "in_progress", "activeForm": label}]},
    }


def test_extract_writes_tasks_from_real_todowrite(fake_claude):
    events = [
        _user("p1", "2026-06-01T10:00:00.000Z", "start the work"),
        _assistant(
            "r1",
            "2026-06-01T10:00:05.000Z",
            "u-p1",
            [{"type": "text", "text": "ok"}, _todowrite("Build the export")],
        ),
        _user("p2", "2026-06-01T10:05:00.000Z", "keep going"),
        _assistant("r2", "2026-06-01T10:05:05.000Z", "u-p2", [{"type": "text", "text": "done"}]),
    ]
    fake_claude.write("todo_session.jsonl", events)
    run_extract(fake_claude.out)

    tasks = _read_csv(fake_claude.out / "tasks.csv")
    links = _read_csv(fake_claude.out / "task_prompts.csv")
    assert len(tasks) == 1
    assert tasks[0]["origin"] == "todo"
    assert tasks[0]["name"] == "Build the export"
    assert tasks[0]["prompts"] == "2"
    assert {e["prompt_id"] for e in links} == {"p1", "p2"}
    assert all(e["task_id"] == tasks[0]["task_id"] for e in links)


def test_task_cost_reconciles_with_total_prompt_cost():
    """Every real prompt is in exactly one task, so the task costs sum to the
    real-prompt bill exactly (the B2 reconciliation invariant on demo data)."""
    ds = dataset_from_csvs(Path("demo_data"))
    assert ds.tasks and ds.task_prompts

    real = {p["prompt_id"] for p in ds.prompts}
    covered = [link["prompt_id"] for link in ds.task_prompts]
    # Coverage: each real prompt mapped once, no edge to a pseudo/overhead prompt.
    assert sorted(covered) == sorted(real)

    engine = CostEngine("anthropic", ds.pricing_path)

    def _cost(prompt_ids: set[str]) -> float:
        return sum(
            engine.cost(row.get("model") or "", row["token_type"], int(row["token_count"]))
            for row in ds.tokens
            if row["prompt_id"] in prompt_ids
        )

    task_to_prompts: dict[str, set[str]] = {}
    for link in ds.task_prompts:
        task_to_prompts.setdefault(link["task_id"], set()).add(link["prompt_id"])

    total_via_tasks = sum(_cost(pids) for pids in task_to_prompts.values())
    total_real = _cost(real)
    assert abs(total_via_tasks - total_real) < 1e-9
    assert total_real > 0
