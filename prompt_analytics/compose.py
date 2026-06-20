"""Output composition metrics (Axe C): what the assistant actually produced.

This module turns assistant ``tool_use`` blocks into **metrics only** -- it
never persists a byte of the source code Claude wrote. From a file-editing tool
call it derives:

* the **language** (mapped from the ``file_path`` extension),
* the **kind** (``code`` vs ``test``, decided by path/name conventions),
* the **lines added / deleted** (an exact LCS line diff over
  ``old_string``/``new_string`` or ``content``).

It also splits a message's output into a **prose** weight (its ``text`` blocks)
and a **code** weight (its ``tool_use`` blocks) using the local tokenizer; the
extractor prorates the real ``output_tokens`` by these weights.

All inputs (file contents, edit strings) are consumed transiently here to count
lines and tokens; only the integer metrics and a project-relative path identity
(needed to count *distinct* files, never written to any CSV) leave this module.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import PurePosixPath
from typing import Any

from .schema import OutputFileRow, ToolEdit
from .tokenizer import count_tokens

__all__ = [
    "detect_language",
    "detect_kind",
    "relativize",
    "diff_lines",
    "serialize_tool_input",
    "tool_edit",
    "analyze_assistant_content",
    "aggregate_output_files",
]

# Tool names that modify files on disk (Claude Code's edit surface). Read/Bash/
# Grep/Glob produce no file row, but their serialized input still counts as
# "code" tokens in the prose/code split.
_EDIT_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})

# Extension -> language label. Unmapped extensions fall back to the bare
# extension (see :func:`detect_language`); the goal is a readable mix, not an
# exhaustive linguist clone.
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "Python",
    ".pyi": "Python",
    ".ipynb": "Jupyter Notebook",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".swift": "Swift",
    ".scala": "Scala",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".ps1": "PowerShell",
    ".sql": "SQL",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "CSS",
    ".sass": "CSS",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".rst": "reStructuredText",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".ini": "INI",
    ".cfg": "INI",
    ".xml": "XML",
    ".txt": "Text",
    ".csv": "CSV",
    ".tf": "Terraform",
    ".lua": "Lua",
    ".r": "R",
    ".dart": "Dart",
    ".ex": "Elixir",
    ".exs": "Elixir",
}

# Directory names that mark a test tree.
_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs", "testing"})

# Filename conventions for test files across ecosystems (operate on the lower-
# cased basename): test_foo.py, foo_test.go, foo.test.ts, foo.spec.ts,
# foo_spec.rb, FooTest.java / FooTests.cs.
_TEST_NAME_RE = re.compile(
    r"^test[_-].+"
    r"|.+[_-]test\.[a-z0-9]+$"
    r"|.+\.(test|spec)\.[a-z0-9]+$"
    r"|.+[_-]spec\.[a-z0-9]+$"
    r"|.+tests?\.(java|kt|cs|scala)$"
)

# Extensionless filenames with a well-known language.
_SPECIAL_NAMES: dict[str, str] = {
    "dockerfile": "Dockerfile",
    "makefile": "Makefile",
}


def _posix(path: str) -> PurePosixPath:
    """Normalize a path (Windows or POSIX) to a POSIX pure path."""
    return PurePosixPath(path.replace("\\", "/"))


def detect_language(path: str) -> str:
    """Map a file path to a language label from its extension.

    Unmapped extensions degrade to the bare extension (e.g. ``.zig`` ->
    ``"zig"``); extensionless files fall back to a few well-known names, else
    ``"other"``.
    """
    name = _posix(path).name.lower()
    if name in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[name]
    suffix = _posix(name).suffix
    if suffix in LANGUAGE_BY_EXT:
        return LANGUAGE_BY_EXT[suffix]
    if suffix:
        return suffix[1:]
    return "other"


def detect_kind(path: str) -> str:
    """Classify a file path as ``"test"`` or ``"code"`` (the only two kinds).

    A path is a test when it lives under a test directory (``tests/``,
    ``__tests__/``, ...) or its basename matches a test naming convention
    (``test_*``, ``*_test.*``, ``*.spec.*``, ``FooTest.java``, ...).
    """
    posix = _posix(path)
    dirs = {part.lower() for part in posix.parts[:-1]}
    if dirs & _TEST_DIRS:
        return "test"
    if _TEST_NAME_RE.match(posix.name.lower()):
        return "test"
    return "code"


def diff_lines(old: str, new: str) -> tuple[int, int]:
    """Exact line diff (added, deleted) between two texts via LCS opcodes.

    A pure creation (``old == ""``) counts every new line as added; this is how
    ``Write`` is measured. Trailing-newline-only differences add/remove nothing.
    """
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    added = deleted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            deleted += i2 - i1
            added += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "insert":
            added += j2 - j1
    return added, deleted


def serialize_tool_input(value: Any) -> str:
    """Serialize a tool ``input`` to a string for *token counting only*.

    The serialized form is never persisted; it exists so the prose/code split
    can weigh a ``tool_use`` block by the tokens the model spent emitting it.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def relativize(path: str, cwd: str) -> str:
    """Project-relative path identity (basename if outside the project tree).

    The file identity that joins Axe C (edits) to Axe D (reads) in the unified
    per-file view. Kept relative so no absolute machine path is ever persisted
    (DASH4 / D5-D6). Shared by ``tool_edit`` (C) and ``context`` (D).
    """
    norm = _posix(path)
    if cwd:
        base = _posix(cwd)
        try:
            return norm.relative_to(base).as_posix()
        except ValueError:
            pass
    return norm.name


def tool_edit(tool_id: str, name: str, raw_input: Any, cwd: str) -> ToolEdit | None:
    """Derive file-edit metrics from one ``tool_use`` block, or ``None``.

    Returns ``None`` for non-editing tools (Read/Bash/...) or malformed inputs.
    Lines come from an exact diff; the path is kept only as a relative identity
    for distinct-file counting (the language/kind are derived from it eagerly).
    """
    if name not in _EDIT_TOOLS or not isinstance(raw_input, dict):
        return None
    file_path = raw_input.get("file_path") or raw_input.get("notebook_path")
    if not isinstance(file_path, str) or not file_path:
        return None

    if name == "Write":
        content = raw_input.get("content")
        added, deleted = diff_lines("", content if isinstance(content, str) else "")
    elif name == "Edit":
        added, deleted = diff_lines(
            str(raw_input.get("old_string", "")), str(raw_input.get("new_string", ""))
        )
    elif name == "MultiEdit":
        added = deleted = 0
        for item in raw_input.get("edits") or []:
            if isinstance(item, dict):
                a, d = diff_lines(str(item.get("old_string", "")), str(item.get("new_string", "")))
                added += a
                deleted += d
    else:  # NotebookEdit
        new_source = raw_input.get("new_source")
        added, deleted = diff_lines("", new_source if isinstance(new_source, str) else "")

    return ToolEdit(
        tool_id=tool_id,
        path=relativize(file_path, cwd),
        language=detect_language(file_path),
        kind=detect_kind(file_path),
        lines_added=added,
        lines_deleted=deleted,
    )


def analyze_assistant_content(content: Any, cwd: str) -> tuple[int, int, list[ToolEdit]]:
    """Split one assistant message's content into output-composition metrics.

    Returns ``(prose_tokens, code_tokens, file_edits)``:

    * ``prose_tokens`` -- local token count of the concatenated ``text`` blocks,
    * ``code_tokens`` -- local token count of every ``tool_use`` block's input,
    * ``file_edits`` -- the metrics for file-editing tool calls (others omitted).

    The token counts weigh the prose/code prorating of the real output tokens;
    no content is returned, only integers and the per-edit metric rows.
    """
    if isinstance(content, str):
        return count_tokens(content), 0, []
    if not isinstance(content, list):
        return 0, 0, []

    prose_parts: list[str] = []
    code_tokens = 0
    edits: list[ToolEdit] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            prose_parts.append(str(block.get("text", "")))
        elif btype == "tool_use":
            code_tokens += count_tokens(serialize_tool_input(block.get("input")))
            tool_id = str(block.get("id") or "")
            edit = tool_edit(tool_id, str(block.get("name") or ""), block.get("input"), cwd)
            if edit is not None and tool_id:
                edits.append(edit)

    prose_tokens = count_tokens("\n".join(prose_parts)) if prose_parts else 0
    return prose_tokens, code_tokens, edits


def aggregate_output_files(prompt_id: str, edits: list[ToolEdit]) -> list[OutputFileRow]:
    """Collapse a prompt's edits into one long row per **file** (path).

    Re-editing one file in the prompt stays one row; ``edits`` counts those edit
    tool calls and the line counts sum across them. ``language``/``kind`` come
    from the path (stable per file). The relative path is the identity that joins
    to Axe D's reads in the unified per-file view. Rows are sorted for a stable
    CSV; metrics only -- no source code.
    """
    groups: dict[str, tuple[str, str, list[int]]] = {}
    for edit in edits:
        _lang, _kind, totals = groups.setdefault(
            edit["path"], (edit["language"], edit["kind"], [0, 0, 0])
        )
        totals[0] += 1  # one edit tool call
        totals[1] += edit["lines_added"]
        totals[2] += edit["lines_deleted"]

    rows = [
        OutputFileRow(
            prompt_id=prompt_id,
            path=path,
            language=language,
            kind=kind,
            edits=totals[0],
            lines_added=totals[1],
            lines_deleted=totals[2],
        )
        for path, (language, kind, totals) in groups.items()
    ]
    rows.sort(key=lambda r: r["path"])
    return rows
