"""Tests for prompt_analytics.snapshot â€” 11.3.

Covers: credentials absent/corrupted, 401, timeout, 5xx mock via urllib,
run_snapshot non-blocking (never raises), append on empty file.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prompt_analytics.snapshot import (
    append_csv,
    parse_rows,
    read_access_token,
    run_snapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TOKEN = "tok-test-abc"

SAMPLE_RESPONSE = {
    "five_hour": {"utilization": 12.3, "resets_at": "R1"},
    "seven_day": {"utilization": 50, "resets_at": "R2"},
    "extra_usage": {"x": 1},
    "weird": {"nope": True},
}


def _make_http_error(code: int, reason: str = "Error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x",
        code=code,
        msg=reason,
        hdrs=MagicMock(),
        fp=io.BytesIO(b""),
    )


def _make_creds(tmp_path: Path) -> None:
    """Write a valid credentials file in tmp_path."""
    creds_dir = tmp_path / ".claude"
    creds_dir.mkdir(parents=True, exist_ok=True)
    (creds_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": SAMPLE_TOKEN}}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# read_access_token
# ---------------------------------------------------------------------------


def test_read_token_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "nonexistent"))
    assert read_access_token() is None


def test_read_token_corrupted_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    creds_dir = tmp_path / ".claude"
    creds_dir.mkdir()
    (creds_dir / ".credentials.json").write_text("{invalid json", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert read_access_token() is None


def test_read_token_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    creds_dir = tmp_path / ".claude"
    creds_dir.mkdir()
    (creds_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert read_access_token() is None


def test_read_token_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    creds_dir = tmp_path / ".claude"
    creds_dir.mkdir()
    (creds_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "abc-123"}}), encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert read_access_token() == "abc-123"


# ---------------------------------------------------------------------------
# parse_rows
# ---------------------------------------------------------------------------


def test_parse_rows_from_api_response() -> None:
    rows = parse_rows(SAMPLE_RESPONSE, "TS")
    assert len(rows) == 2
    assert {r["field"] for r in rows} == {"five_hour", "seven_day"}
    for r in rows:
        assert set(r.keys()) == {"snapshot_at", "field", "utilization_pct", "resets_at"}
        assert r["snapshot_at"] == "TS"


def test_parse_rows_empty_response() -> None:
    assert parse_rows({}, "TS") == []


# ---------------------------------------------------------------------------
# append_csv
# ---------------------------------------------------------------------------


def test_append_csv_creates_header_on_first_run(tmp_path: Path) -> None:
    path = tmp_path / "quota_log.csv"
    append_csv(
        [{"snapshot_at": "TS", "field": "f", "utilization_pct": 1.0, "resets_at": "R"}],
        path,
    )
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "snapshot_at,field,utilization_pct,resets_at"


def test_append_csv_header_on_empty_file(tmp_path: Path) -> None:
    """Header must be written even when the file exists but is empty (B5)."""
    path = tmp_path / "quota_log.csv"
    path.write_text("", encoding="utf-8")
    append_csv(
        [{"snapshot_at": "TS", "field": "f", "utilization_pct": 1.0, "resets_at": "R"}],
        path,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "snapshot_at,field,utilization_pct,resets_at"
    assert len(lines) == 2


def test_append_csv_no_header_on_subsequent_runs(tmp_path: Path) -> None:
    path = tmp_path / "quota_log.csv"
    row = {"snapshot_at": "TS", "field": "f", "utilization_pct": 1.0, "resets_at": "R"}
    append_csv([row], path)
    append_csv([row], path)
    content = path.read_text(encoding="utf-8")
    assert content.count("snapshot_at") == 1


# ---------------------------------------------------------------------------
# run_snapshot â€” non-blocking on every failure mode
# ---------------------------------------------------------------------------


def test_run_snapshot_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = run_snapshot(tmp_path / "out")
    assert result == 0
    assert "no Claude access token" in capsys.readouterr().err


def test_run_snapshot_401_prints_refresh_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_creds(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with patch("urllib.request.urlopen", side_effect=_make_http_error(401)):
        result = run_snapshot(tmp_path / "out")
    assert result == 0
    err = capsys.readouterr().err
    assert "401" in err
    assert "Restart Claude Code" in err


def test_run_snapshot_5xx_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_creds(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with patch("urllib.request.urlopen", side_effect=_make_http_error(503, "Service Unavailable")):
        result = run_snapshot(tmp_path / "out")
    assert result == 0
    assert "503" in capsys.readouterr().err


def test_run_snapshot_timeout_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_creds(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        result = run_snapshot(tmp_path / "out")
    assert result == 0
    assert "quota snapshot failed" in capsys.readouterr().err


def test_run_snapshot_url_error_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_creds(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("name not resolved"),
    ):
        result = run_snapshot(tmp_path / "out")
    assert result == 0
    assert "quota snapshot failed" in capsys.readouterr().err


def test_run_snapshot_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_creds(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    body = json.dumps(SAMPLE_RESPONSE).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = body
    with patch("urllib.request.urlopen", return_value=mock_resp):
        out_dir = tmp_path / "out"
        result = run_snapshot(out_dir)
    assert result == 2
    csv_path = out_dir / "quota_log.csv"
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "snapshot_at,field,utilization_pct,resets_at"
    assert len(lines) == 3  # header + 2 rows


def test_run_snapshot_does_not_raise_on_corrupted_creds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_snapshot must never propagate exceptions."""
    creds_dir = tmp_path / ".claude"
    creds_dir.mkdir()
    (creds_dir / ".credentials.json").write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = run_snapshot(tmp_path / "out")
    assert result == 0


def test_run_snapshot_empty_api_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty (no quota fields) response should warn and return 0."""
    _make_creds(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    body = json.dumps({"extra_usage": {}}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = body
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = run_snapshot(tmp_path / "out")
    assert result == 0
    assert "no quota fields" in capsys.readouterr().err
