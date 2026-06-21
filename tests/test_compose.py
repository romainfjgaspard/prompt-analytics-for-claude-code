"""Unit tests for the Axe C output-composition helpers (compose.py).

Language, kind and line diffs are *exact* (no tokenizer involved); the
prose/code token weighting is checked with a stubbed deterministic counter so
the assertion does not depend on tiktoken or the network.
"""

from __future__ import annotations

import pytest

from prompt_analytics import compose
from prompt_analytics.schema import ToolEdit


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/parser.py", "Python"),
        ("/abs/win\\style\\Module.TS", "TypeScript"),
        ("app/main.go", "Go"),
        ("notebook.ipynb", "Jupyter Notebook"),
        ("Dockerfile", "Dockerfile"),
        ("Makefile", "Makefile"),
        ("weird.zig", "zig"),  # unmapped -> bare extension
        ("LICENSE", "other"),  # no extension, unknown name
    ],
)
def test_detect_language(path, expected):
    assert compose.detect_language(path) == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/parser.py", "code"),
        ("tests/test_parser.py", "test"),
        ("pkg/foo_test.go", "test"),
        ("web/Button.test.tsx", "test"),
        ("web/Button.spec.ts", "test"),
        ("spec/user_spec.rb", "test"),
        ("src/UserTest.java", "test"),
        ("__tests__/widget.js", "test"),
        ("src/latest.py", "code"),  # 'latest' must not look like a test
        ("README.md", "code"),
    ],
)
def test_detect_kind(path, expected):
    assert compose.detect_kind(path) == expected


def test_diff_lines_creation_counts_every_line():
    assert compose.diff_lines("", "a\nb\nc") == (3, 0)


def test_diff_lines_replace_and_insert():
    added, deleted = compose.diff_lines(
        "def parse(x):\n    return x",
        "def parse(x):\n    return int(x)\n    # extra",
    )
    assert (added, deleted) == (2, 1)


def test_diff_lines_pure_deletion():
    assert compose.diff_lines("a\nb\nc", "a") == (0, 2)


def test_tool_edit_write_is_all_added():
    edit = compose.tool_edit(
        "t1", "Write", {"file_path": "/proj/src/m.py", "content": "x\ny"}, "/proj"
    )
    assert edit is not None
    assert edit["language"] == "Python"
    assert edit["kind"] == "code"
    assert edit["path"] == "src/m.py"  # relative to cwd
    assert (edit["lines_added"], edit["lines_deleted"]) == (2, 0)


def test_tool_edit_multiedit_sums_each_hunk():
    edit = compose.tool_edit(
        "t2",
        "MultiEdit",
        {
            "file_path": "/proj/a.py",
            "edits": [
                {"old_string": "a", "new_string": "a\nb"},  # +1
                {"old_string": "c\nd", "new_string": "c"},  # -1
            ],
        },
        "/proj",
    )
    assert edit is not None
    assert (edit["lines_added"], edit["lines_deleted"]) == (1, 1)


def test_tool_edit_ignores_non_editing_tools_and_pathless_inputs():
    assert compose.tool_edit("t3", "Read", {"file_path": "/proj/a.py"}, "/proj") is None
    assert compose.tool_edit("t4", "Bash", {"command": "ls"}, "/proj") is None
    assert compose.tool_edit("t5", "Write", {"content": "x"}, "/proj") is None


def test_tool_edit_outside_project_falls_back_to_basename():
    edit = compose.tool_edit(
        "t6", "Write", {"file_path": "/elsewhere/x.py", "content": "y"}, "/proj"
    )
    assert edit is not None
    assert edit["path"] == "x.py"  # no absolute machine path leaks


def test_analyze_splits_prose_and_code(monkeypatch):
    # Deterministic counter: one "token" per character.
    monkeypatch.setattr(compose, "count_tokens", len)
    content = [
        {"type": "text", "text": "abcd"},  # 4 prose chars
        {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"x": 1}},
        {
            "type": "tool_use",
            "id": "tu2",
            "name": "Write",
            "input": {"file_path": "/p/a.py", "content": "z"},
        },
    ]
    prose, code, edits = compose.analyze_assistant_content(content, "/p")
    assert prose == 4
    assert code > 0  # both tool_use inputs counted
    # Only the file-editing tool yields an edit row.
    assert [e["tool_id"] for e in edits] == ["tu2"]


def test_analyze_handles_string_content(monkeypatch):
    monkeypatch.setattr(compose, "count_tokens", len)
    prose, code, edits = compose.analyze_assistant_content("hello", "/p")
    assert (prose, code, edits) == (5, 0, [])


def test_aggregate_output_files_counts_distinct_files():
    edits = [
        ToolEdit(
            tool_id="a",
            path="src/x.py",
            language="Python",
            kind="code",
            lines_added=5,
            lines_deleted=0,
        ),
        ToolEdit(
            tool_id="b",
            path="src/x.py",
            language="Python",
            kind="code",
            lines_added=2,
            lines_deleted=1,
        ),
        ToolEdit(
            tool_id="c",
            path="tests/t.py",
            language="Python",
            kind="test",
            lines_added=3,
            lines_deleted=0,
        ),
    ]
    rows = compose.aggregate_output_files("p1", edits)
    by_path = {r["path"]: r for r in rows}
    # x.py edited twice -> one row, edits counted, lines summed.
    assert by_path["src/x.py"]["edits"] == 2
    assert by_path["src/x.py"]["kind"] == "code"
    assert (by_path["src/x.py"]["lines_added"], by_path["src/x.py"]["lines_deleted"]) == (7, 1)
    assert by_path["tests/t.py"]["edits"] == 1
    assert by_path["tests/t.py"]["kind"] == "test"
    assert all(r["prompt_id"] == "p1" for r in rows)
