"""Tests for the atomic write helpers."""

from __future__ import annotations

import csv

import pytest

from prompt_analytics.storage import (
    append_csv,
    atomic_write_csv,
    atomic_write_json,
    escape_csv_formula,
)

COLS = ["a", "b"]
ROWS = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_atomic_write_csv_roundtrip(tmp_path):
    path = tmp_path / "out.csv"
    atomic_write_csv(path, COLS, ROWS)
    assert _read(path) == ROWS
    # Overwrite works and leaves no temp files behind.
    atomic_write_csv(path, COLS, ROWS[:1])
    assert _read(path) == ROWS[:1]
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_csv_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "out.csv"
    atomic_write_csv(path, COLS, ROWS)
    assert _read(path) == ROWS


def test_atomic_write_keeps_previous_file_on_failure(tmp_path):
    path = tmp_path / "out.csv"
    atomic_write_csv(path, COLS, ROWS)

    class Exploding:
        """Mapping whose iteration fails mid-write."""

        def keys(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_write_csv(path, COLS, [Exploding()])  # type: ignore[list-item]
    # The original content is intact and no temp file lingers.
    assert _read(path) == ROWS
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_json(tmp_path):
    path = tmp_path / "cache.json"
    atomic_write_json(path, {"k": [1, 2]})
    import json

    assert json.loads(path.read_text(encoding="utf-8")) == {"k": [1, 2]}


def test_append_csv_writes_header_when_absent(tmp_path):
    path = tmp_path / "log.csv"
    append_csv(path, COLS, ROWS[:1])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "a,b"
    assert len(lines) == 2


def test_append_csv_writes_header_on_empty_file(tmp_path):
    """B5 regression: an existing 0-byte file must still get a header."""
    path = tmp_path / "log.csv"
    path.touch()
    append_csv(path, COLS, ROWS[:1])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "a,b"


def test_append_csv_no_duplicate_header(tmp_path):
    path = tmp_path / "log.csv"
    append_csv(path, COLS, ROWS[:1])
    append_csv(path, COLS, ROWS[1:])
    content = path.read_text(encoding="utf-8")
    assert content.count("a,b") == 1
    assert len(_read(path)) == 2


# ---------------------------------------------------------------------------
# CSV formula injection (10.2 / R9c).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("=cmd()", "'=cmd()"),
        ("+1+1", "'+1+1"),
        ("-2+3", "'-2+3"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\trm -rf", "'\trm -rf"),
        ("\rfoo", "'\rfoo"),
        ("normal text", "normal text"),
        ("", ""),
        (None, None),
        (5, 5),
    ],
)
def test_escape_csv_formula(raw, expected):
    assert escape_csv_formula(raw) == expected


def test_atomic_write_csv_escapes_only_risk_columns(tmp_path):
    """A formula-leading prompt_preview is neutralized; numeric cols stay raw."""
    path = tmp_path / "prompts.csv"
    cols = ["prompt_id", "char_count", "prompt_preview", "prompt_text"]
    rows = [
        {
            "prompt_id": "p1",
            "char_count": "-5",  # not a risk column: must stay verbatim
            "prompt_preview": "=HYPERLINK(0)",
            "prompt_text": "@evil",
        }
    ]
    atomic_write_csv(path, cols, rows)
    out = _read(path)[0]
    assert out["char_count"] == "-5"
    assert out["prompt_preview"] == "'=HYPERLINK(0)"
    assert out["prompt_text"] == "'@evil"
