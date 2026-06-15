"""Capture an anonymized parsing fixture from a real Claude Code JSONL log.

Per the ccusage maintainers' advice ("pin your parsing against fixture files
per version"), we keep one fixture directory per JSONL *format* version under
``tests/fixtures/claude-code-<version>/``. This script turns one of your own
``~/.claude/projects/**/*.jsonl`` files into such a fixture, scrubbing the
content while keeping everything the parser counts on.

What is preserved verbatim (so token/cost totals stay reproducible):

* the line structure, every key, and the nesting;
* ids and the attribution chain (``uuid``, ``parentUuid``, ``requestId``,
  ``promptId``, ``sessionId``, ``message.id``, tool-use ids);
* ``message.usage`` (all token counts), ``model``, ``timestamp``,
  ``stop_reason``, ``version`` and the boolean flags (``isMeta``,
  ``isSidechain``, ``isCompactSummary`` …);
* the literal markers the parser filters on (``<command-name>``,
  ``[Request interrupted by user`` …), spliced back at their original offsets.

What is anonymized:

* every other free-text string (prompts, assistant text, thinking, tool
  inputs/outputs, titles) — replaced **character for character** (letters → x,
  digits → 0, punctuation/whitespace kept), so lengths and ``char_count`` are
  preserved but no real content survives;
* ``cwd`` paths (mapped to a generic ``project-N`` path) and ``gitBranch``.

Usage::

    python scripts/capture_fixture.py path/to/session.jsonl
    python scripts/capture_fixture.py path/to/session.jsonl --version 2.1.173

Always eyeball the produced fixture before committing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Keys whose values must survive untouched for parsing/counting to match the
# original. Anything under one of these (e.g. the whole ``usage`` mapping) is
# copied verbatim.
VERBATIM_KEYS = frozenset(
    {
        "type",
        "role",
        "model",
        "stop_reason",
        "stop_sequence",
        "id",
        "uuid",
        "parentUuid",
        "requestId",
        "promptId",
        "sessionId",
        "leafUuid",
        "messageId",
        "tool_use_id",
        "version",
        "timestamp",
        "userType",
        "entrypoint",
        "permissionMode",
        "promptSource",
        "usage",
        "isSidechain",
        "isMeta",
        "isCompactSummary",
        "isApiErrorMessage",
        "isSnapshotUpdate",
        "isVisibleInTranscriptOnly",
    }
)

# Literal substrings the extractor keys filtering decisions on (see
# ``extract.prompt_skip_reason``). They are restored after anonymization so a
# captured fixture filters exactly like the original did.
PRESERVED_MARKERS = (
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "[Request interrupted by user",
    "This session is being continued from a previous conversation",
)


def anonymize_text(text: str) -> str:
    """Replace content character-for-character, keeping length and structure.

    Letters become ``x``, digits ``0``; whitespace and punctuation are kept so
    multi-line blocks stay multi-line. Known filter markers are spliced back at
    their exact original positions (lengths are identical, so offsets align).
    """
    chars = []
    for ch in text:
        if ch.isalpha():
            chars.append("x")
        elif ch.isdigit():
            chars.append("0")
        else:
            chars.append(ch)
    out = "".join(chars)
    for marker in PRESERVED_MARKERS:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            out = out[:idx] + marker + out[idx + len(marker) :]
            start = idx + len(marker)
    return out


class _CwdMapper:
    """Map each distinct real ``cwd`` to a stable, generic fake path."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}

    def fake(self, original: str) -> str:
        if original not in self._seen:
            n = len(self._seen) + 1
            if "\\" in original and "/" not in original:
                fake = f"C:\\Users\\user\\Code\\project-{n}"
            else:
                fake = f"/home/user/code/project-{n}"
            self._seen[original] = fake
        return self._seen[original]


def anonymize(value: Any, mapper: _CwdMapper, key: str | None = None) -> Any:
    """Recursively anonymize a parsed JSON value (see module docstring)."""
    if key in VERBATIM_KEYS:
        return value
    if key == "cwd" and isinstance(value, str):
        return mapper.fake(value)
    if key == "gitBranch" and isinstance(value, str):
        return "main" if value else value
    if isinstance(value, dict):
        return {k: anonymize(v, mapper, k) for k, v in value.items()}
    if isinstance(value, list):
        return [anonymize(v, mapper, key) for v in value]
    if isinstance(value, str):
        return anonymize_text(value)
    return value


def _detect_version(lines: list[str]) -> str | None:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        version = obj.get("version")
        if isinstance(version, str) and version:
            return version
    return None


def capture(
    input_path: Path,
    fixtures_root: Path,
    version: str | None,
    project: str,
) -> Path:
    """Anonymize ``input_path`` into the versioned fixtures tree; return the path."""
    raw = input_path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    version = version or _detect_version(lines)
    if not version:
        raise SystemExit("Could not detect a Claude Code version in the input; pass --version.")

    mapper = _CwdMapper()
    out_lines: list[str] = []
    skipped = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            skipped += 1
            continue
        anon = anonymize(obj, mapper)
        out_lines.append(json.dumps(anon, ensure_ascii=False, separators=(",", ":")))

    out_dir = fixtures_root / f"claude-code-{version}" / project
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}.jsonl"
    # newline="\n": a fixture captured on Windows must not come out CRLF when
    # the committed ones are LF (08 m3).
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(out_lines) + "\n")

    print(f"Captured {len(out_lines)} lines (skipped {skipped} unparseable)")
    print(f"  version : {version}")
    print(f"  projects: {len(mapper._seen)} distinct cwd -> project-N")
    print(f"  written : {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("input", type=Path, help="Path to a real .jsonl session log.")
    parser.add_argument(
        "--version",
        help="Claude Code version label (default: auto-detected from the log).",
    )
    parser.add_argument(
        "--project",
        default="demo-project",
        help="Generic project folder name for the fixture (default: %(default)s).",
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures",
        help="Root of the fixtures tree (default: tests/fixtures).",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"No such file: {args.input}", file=sys.stderr)
        return 2
    capture(args.input, args.fixtures_root, args.version, args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
