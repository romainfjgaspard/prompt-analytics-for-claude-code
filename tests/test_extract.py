"""Tests for the extraction pipeline (counting rules, robustness, report)."""

from __future__ import annotations

import csv
import os

import pytest

from prompt_analytics.compose import analyze_assistant_content
from prompt_analytics.extract import run_extract
from prompt_analytics.tokenizer import count_tokens


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _tokens_by_prompt(out):
    """{prompt_id: {token_type: count}} from tokens.csv (summed across models)."""
    result: dict[str, dict[str, int]] = {}
    for row in _read_csv(out / "tokens.csv"):
        counts = result.setdefault(row["prompt_id"], {})
        counts[row["token_type"]] = counts.get(row["token_type"], 0) + int(row["token_count"])
    return result


def _user(pid, ts, text, sid="sess-w", uuid=None, parent=None, **extra):
    event = {
        "type": "user",
        "promptId": pid,
        "uuid": uuid or f"u-{pid}",
        "parentUuid": parent,
        "timestamp": ts,
        "cwd": "/home/fake/projects/written",
        "gitBranch": "main",
        "entrypoint": "cli",
        "version": "2.1.169",
        "sessionId": sid,
        "message": {"role": "user", "content": text},
    }
    event.update(extra)
    return event


def _assistant(req, ts, sid="sess-w", uuid=None, parent=None, model="claude-sonnet-4-6", **usage):
    tokens = {
        "input_tokens": usage.pop("inp", 0),
        "output_tokens": usage.pop("out", 0),
        "cache_read_input_tokens": usage.pop("cache_read", 0),
        "cache_creation_input_tokens": usage.pop("cache_write", 0),
    }
    event = {
        "type": "assistant",
        "uuid": uuid or f"u-{req}",
        "parentUuid": parent,
        "requestId": req,
        "timestamp": ts,
        "sessionId": sid,
        "cwd": "/home/fake/projects/written",
        "message": {
            "id": f"m-{req}",
            "role": "assistant",
            "model": model,
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            "usage": tokens,
        },
    }
    event.update(usage)
    return event


# ---------------------------------------------------------------------------
# Happy path + legacy-format fixtures.
# ---------------------------------------------------------------------------


def test_full_extract_produces_all_csvs(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    fake_claude.add("session_beta.jsonl", project="beta")
    fake_claude.add("session_gamma.jsonl", project="gamma")

    report = run_extract(fake_claude.out)

    out = fake_claude.out
    for name in (
        "sessions.csv",
        "prompts.csv",
        "tokens.csv",
        "requests.csv",
        "token_types.csv",
        "prompts_text.csv",
        "context_sources.csv",
        "context_cost.csv",
        "extract_meta.json",
    ):
        assert (out / name).exists(), f"missing {name}"

    assert len(_read_csv(out / "sessions.csv")) == 3
    assert len(_read_csv(out / "prompts.csv")) == 5
    # Only non-zero (prompt, token_type) pairs are written.
    assert len(_read_csv(out / "tokens.csv")) == 16
    assert report.sessions == 3
    assert report.prompts == 5
    assert report.exit_code() == 0


def test_requestid_deduplication_keeps_latest_snapshot(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    report = run_extract(fake_claude.out)

    pa1 = _tokens_by_prompt(fake_claude.out)["pA1"]
    # rA1a is written twice with progressive usage (out 50 then 60): the
    # message is counted once, with its LATEST snapshot.
    assert pa1["input"] == 130
    assert pa1["output"] == 100
    assert pa1["cache_read"] == 210
    # Legacy total falls back to the 5m bucket.
    assert pa1["cache_write_5m"] == 25
    assert report.deduplicated_records == 1


def test_growing_session_between_runs(fake_claude):
    """A session file that gains prompts between two runs is fully re-read.

    Anti-regression for B1: the old incremental mode froze any session it had
    already seen, silently losing every later prompt.
    """
    events = [
        _user("pG1", "2026-06-03T10:00:00.000Z", "first prompt"),
        _assistant("rG1", "2026-06-03T10:00:05.000Z", parent="u-pG1", inp=10, out=20),
    ]
    path = fake_claude.write("grow.jsonl", events)
    run_extract(fake_claude.out)
    assert [r["prompt_id"] for r in _read_csv(fake_claude.out / "prompts.csv")] == ["pG1"]

    events += [
        _user("pG2", "2026-06-03T11:00:00.000Z", "second prompt", parent="u-rG1"),
        _assistant("rG2", "2026-06-03T11:00:05.000Z", parent="u-pG2", inp=30, out=40),
    ]
    path = fake_claude.write("grow.jsonl", events)
    os.utime(path, (path.stat().st_atime + 10, path.stat().st_mtime + 10))

    run_extract(fake_claude.out)
    prompts = _read_csv(fake_claude.out / "prompts.csv")
    assert [r["prompt_id"] for r in prompts] == ["pG1", "pG2"]
    tokens = _tokens_by_prompt(fake_claude.out)
    # No double counting of the replayed first turn either.
    assert tokens["pG1"] == {"input": 10, "output": 20}
    assert tokens["pG2"] == {"input": 30, "output": 40}


def test_parse_cache_hit_and_invalidation(fake_claude):
    path = fake_claude.add("session_alpha.jsonl", project="alpha")

    first = run_extract(fake_claude.out)
    assert first.files_cached == 0

    second = run_extract(fake_claude.out)
    assert second.files_cached == second.files_read == 1
    assert second.prompts == first.prompts

    os.utime(path, (path.stat().st_atime + 60, path.stat().st_mtime + 60))
    third = run_extract(fake_claude.out)
    assert third.files_cached == 0
    assert third.prompts == first.prompts

    fourth = run_extract(fake_claude.out, use_cache=False)
    assert fourth.files_cached == 0


# ---------------------------------------------------------------------------
# Counting correctness (3.3 - 3.8).
# ---------------------------------------------------------------------------


def test_cross_file_dedup_resumed_session(fake_claude):
    """A resumed session replays records into a new file; count them once."""
    fake_claude.add("session_delta.jsonl", project="delta")
    fake_claude.add("session_delta_resumed.jsonl", project="delta")

    report = run_extract(fake_claude.out)

    prompts = _read_csv(fake_claude.out / "prompts.csv")
    by_id = {r["prompt_id"]: r for r in prompts}
    # pD5 belongs to the original session; the replayed copy is not a new prompt.
    assert sorted(by_id) == ["pD1", "pD5", "pE1"]
    assert by_id["pD5"]["session_id"] == "sess-ddd"
    assert by_id["pE1"]["session_id"] == "sess-eee"
    assert report.deduplicated_records == 1

    tokens = _tokens_by_prompt(fake_claude.out)
    # Replayed rD5/mD5 not double counted (40 input + 11 from the inline sidechain).
    assert tokens["pD5"]["input"] == 51
    assert tokens["pE1"] == {
        "input": 30,
        "output": 90,
        "cache_read": 50,
        "cache_write_5m": 20,
    }
    session_ids = {r["session_id"] for r in _read_csv(fake_claude.out / "sessions.csv")}
    assert session_ids == {"sess-ddd", "sess-eee"}


def test_attribution_by_prompt_id_chaining(fake_claude):
    """Usage is attributed via promptId / parentUuid chain, across tool results."""
    fake_claude.add("session_delta.jsonl", project="delta")
    run_extract(fake_claude.out)

    tokens = _tokens_by_prompt(fake_claude.out)
    assert tokens["pD1"] == {
        "input": 120,
        "output": 80,
        "cache_read": 600,
        "cache_write_5m": 30,
        "cache_write_1h": 70,
    }
    prompts = {r["prompt_id"]: r for r in _read_csv(fake_claude.out / "prompts.csv")}
    assert prompts["pD1"]["assistant_turns"] == "2"
    assert prompts["pD1"]["tool_calls"] == "1"
    assert prompts["pD1"]["final_stop_reason"] == "end_turn"
    assert prompts["pD1"]["model"] == "claude-opus-4-8"
    assert prompts["pD1"]["mode"] == "plan"
    assert prompts["pD5"]["mode"] == "normal"


def test_fake_prompts_filtered_but_usage_counted(fake_claude):
    """isMeta / command tags / interruptions / compaction are not prompts (3.5)."""
    fake_claude.add("session_delta.jsonl", project="delta")
    report = run_extract(fake_claude.out)

    prompt_ids = {r["prompt_id"] for r in _read_csv(fake_claude.out / "prompts.csv")}
    assert prompt_ids == {"pD1", "pD5"}
    assert report.filtered_prompts == {
        "meta": 1,
        "interrupted": 1,
        "local_command": 1,
        "compact_continuation": 1,
        "sidechain": 1,
    }

    # Their token usage is still counted, attached to the session.
    tokens = _tokens_by_prompt(fake_claude.out)
    assert tokens["pD0"] == {"input": 10, "output": 5, "cache_write_5m": 7}
    assert tokens["pD4"] == {"input": 5, "output": 7, "cache_write_5m": 1000}
    by_session = {
        r["prompt_id"]: r["session_id"] for r in _read_csv(fake_claude.out / "tokens.csv")
    }
    assert by_session["pD0"] == "sess-ddd"
    assert by_session["pD4"] == "sess-ddd"


def test_sidechain_inline_and_subagent_files(fake_claude):
    """Sidechain/subagent cost goes to the parent prompt; turns/tools do not (3.6)."""
    fake_claude.add("session_delta.jsonl", project="delta")
    fake_claude.add_subagent("agent_delta.jsonl", project="delta", parent_session="sess-ddd")

    report = run_extract(fake_claude.out)
    assert report.files_read == 2

    tokens = _tokens_by_prompt(fake_claude.out)
    # pD1 = main turns (120/80) + separate subagent file (200/100, 5m 50).
    assert tokens["pD1"] == {
        "input": 320,
        "output": 180,
        "cache_read": 600,
        "cache_write_5m": 80,
        "cache_write_1h": 70,
    }
    # pD5 = own turn (40/60) + inline sidechain (11/13).
    assert tokens["pD5"]["input"] == 51
    assert tokens["pD5"]["output"] == 73

    prompts = {r["prompt_id"]: r for r in _read_csv(fake_claude.out / "prompts.csv")}
    # Sidechain activity never inflates assistant_turns / tool_calls.
    assert prompts["pD1"]["assistant_turns"] == "2"
    assert prompts["pD1"]["tool_calls"] == "1"
    assert prompts["pD5"]["assistant_turns"] == "1"
    # The subagent's own "prompt" is not a human prompt.
    assert set(prompts) == {"pD1", "pD5"}
    # No extra session was invented for the subagent file.
    session_ids = {r["session_id"] for r in _read_csv(fake_claude.out / "sessions.csv")}
    assert session_ids == {"sess-ddd"}


def test_cache_ttl_granularity_and_server_tool_use(fake_claude):
    fake_claude.add("session_delta.jsonl", project="delta")
    run_extract(fake_claude.out)

    tokens = _tokens_by_prompt(fake_claude.out)
    # 5m and 1h buckets are kept distinct (billed 1.25x vs 2x).
    assert tokens["pD1"]["cache_write_5m"] == 30
    assert tokens["pD1"]["cache_write_1h"] == 70
    # Server-side tool requests are tracked as their own type.
    assert tokens["pD5"]["server_tool_use"] == 2


def test_legacy_cache_total_falls_back_to_5m(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out)
    tokens = _tokens_by_prompt(fake_claude.out)
    assert tokens["pA1"]["cache_write_5m"] == 25
    assert "cache_write_1h" not in tokens["pA1"]


def test_orphan_usage_goes_to_continuation_pseudo_prompt(fake_claude):
    events = [
        _assistant("rX0", "2026-06-03T10:00:00.000Z", sid="sess-x", inp=9, out=4),
        _user("pX1", "2026-06-03T10:01:00.000Z", "real prompt", sid="sess-x"),
        _assistant("rX1", "2026-06-03T10:01:05.000Z", sid="sess-x", parent="u-pX1", inp=1, out=2),
    ]
    fake_claude.write("orphan.jsonl", events)
    run_extract(fake_claude.out)

    tokens = _tokens_by_prompt(fake_claude.out)
    assert tokens["sess-x:_continuation"] == {"input": 9, "output": 4}
    assert tokens["pX1"] == {"input": 1, "output": 2}
    prompt_ids = {r["prompt_id"] for r in _read_csv(fake_claude.out / "prompts.csv")}
    assert prompt_ids == {"pX1"}


# ---------------------------------------------------------------------------
# Request grain (1.1, V10) + sidechain dimension (1.2) + compaction (1.4).
# ---------------------------------------------------------------------------


def test_requests_csv_one_row_per_deduplicated_request(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    report = run_extract(fake_claude.out)

    rows = _read_csv(fake_claude.out / "requests.csv")
    # rA1a is written twice (progressive snapshots): ONE request row, carrying
    # the largest snapshot; rA1b and rA2a follow.
    assert [r["prompt_id"] for r in rows] == ["pA1", "pA1", "pA2"]
    assert [r["request_index"] for r in rows] == ["1", "2", "1"]
    assert report.request_rows == 3

    first = rows[0]
    assert first["input_tokens"] == "100"
    assert first["output_tokens"] == "60"  # the largest snapshot, not the first
    assert first["cache_write_5m_tokens"] == "20"  # legacy total -> 5m bucket
    assert first["model"] == "claude-opus-4-8"
    assert first["stop_reason"] == "tool_use"
    assert first["timestamp"]  # chronologically sortable ISO timestamp
    assert rows[1]["stop_reason"] == "end_turn"


def test_requests_sums_match_tokens_per_prompt_to_the_token(fake_claude):
    """V7: per prompt (pseudo-prompts included), requests.csv column sums
    reproduce tokens.csv exactly -- the request grain refines, never drifts."""
    fake_claude.add("session_alpha.jsonl", project="alpha")
    fake_claude.add("session_delta.jsonl", project="delta")
    fake_claude.add("session_delta_resumed.jsonl", project="delta")
    fake_claude.add_subagent("agent_delta.jsonl", project="delta", parent_session="sess-ddd")
    run_extract(fake_claude.out)

    col_of = {
        "input": "input_tokens",
        "output": "output_tokens",
        "cache_read": "cache_read_tokens",
        "cache_write_5m": "cache_write_5m_tokens",
        "cache_write_1h": "cache_write_1h_tokens",
        "server_tool_use": "server_tool_use_requests",
    }
    request_sums: dict[str, dict[str, int]] = {}
    for row in _read_csv(fake_claude.out / "requests.csv"):
        sums = request_sums.setdefault(row["prompt_id"], dict.fromkeys(col_of, 0))
        for token_type, col in col_of.items():
            sums[token_type] += int(row[col])

    tokens = _tokens_by_prompt(fake_claude.out)
    assert set(request_sums) == set(tokens)
    for pid, counts in tokens.items():
        for token_type, count in counts.items():
            assert request_sums[pid][token_type] == count, (pid, token_type)


def test_tokens_csv_splits_sidechain_as_a_dimension(fake_claude):
    """is_sidechain is first-class in tokens.csv; summing over it reproduces
    the per-prompt totals (V7)."""
    fake_claude.add("session_delta.jsonl", project="delta")
    fake_claude.add_subagent("agent_delta.jsonl", project="delta", parent_session="sess-ddd")
    run_extract(fake_claude.out)

    token_rows = _read_csv(fake_claude.out / "tokens.csv")
    pd1_input = {
        (r["is_sidechain"], r["model"]): int(r["token_count"])
        for r in token_rows
        if r["prompt_id"] == "pD1" and r["token_type"] == "input"
    }
    # Main turns on opus, the subagent file's usage on haiku, kept apart.
    assert pd1_input == {("0", "claude-opus-4-8"): 120, ("1", "claude-haiku-4-5"): 200}
    assert _tokens_by_prompt(fake_claude.out)["pD1"]["input"] == 320

    requests = _read_csv(fake_claude.out / "requests.csv")
    sidechain_pids = {r["prompt_id"] for r in requests if r["is_sidechain"] == "1"}
    # Inline sidechain (pD5) and separate subagent file (pD1).
    assert sidechain_pids == {"pD1", "pD5"}


def test_post_compact_continuation_requests_are_marked(fake_claude):
    """The synthetic post-compaction continuation's usage is flagged (1.4)."""
    fake_claude.add("session_delta.jsonl", project="delta")
    run_extract(fake_claude.out)

    rows = _read_csv(fake_claude.out / "requests.csv")
    flags = {r["prompt_id"]: r["post_compact"] for r in rows}
    # rD4 answers the isCompactSummary message (filtered prompt pD4).
    assert flags["pD4"] == "1"
    # The next real human prompt (pD5) breaks the chain; pD0/pD1 predate it.
    assert flags["pD5"] == "0"
    assert flags["pD0"] == "0"
    assert flags["pD1"] == "0"


# ---------------------------------------------------------------------------
# Idiomatic paths: CLAUDE_CONFIG_DIR + parse-cache GC (1.6, 08 m1/m2).
# ---------------------------------------------------------------------------


def test_claude_config_dir_env_is_honored(fake_claude, tmp_path, monkeypatch):
    """A relocated Claude dir must feed extract too, not just snapshot (m2)."""
    import shutil
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    moved = tmp_path / "moved-claude"
    dest = moved / "projects" / "alpha"
    dest.mkdir(parents=True)
    shutil.copy(fixtures / "session_alpha.jsonl", dest / "session_alpha.jsonl")

    # The default ~/.claude/projects (fake_claude's sandbox) stays empty.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(moved))
    report = run_extract(fake_claude.out)
    assert report.prompts == 2


def test_parse_cache_gc_removes_orphan_entries(fake_claude):
    """Cache entries of deleted JSONL files are dropped, not kept forever."""
    from prompt_analytics import paths

    removed = fake_claude.add("session_alpha.jsonl", project="alpha")
    fake_claude.add("session_beta.jsonl", project="beta")
    run_extract(fake_claude.out)
    cache = paths.parse_cache_dir()
    assert len(list(cache.glob("*.json"))) == 2

    removed.unlink()
    run_extract(fake_claude.out)
    assert len(list(cache.glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# Date filters (3.9).
# ---------------------------------------------------------------------------


def test_date_filter_since_until(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    fake_claude.add("session_beta.jsonl", project="beta")
    fake_claude.add("session_gamma.jsonl", project="gamma")

    report = run_extract(
        fake_claude.out,
        since="2026-05-10",
        until="2026-05-31",
        timezone_name="UTC",
    )

    sessions = _read_csv(fake_claude.out / "sessions.csv")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-bbb"

    prompts = _read_csv(fake_claude.out / "prompts.csv")
    assert sorted(r["prompt_id"] for r in prompts) == ["pB1", "pB2"]
    assert "kept 2 of 5 prompts" in report.window_note


def test_date_filter_applies_at_prompt_grain(fake_claude):
    """A session straddling the bound keeps its in-window prompts (anti-R3)."""
    events = [
        _user("p1", "2026-05-01T10:00:00.000Z", "early prompt"),
        _assistant("r1", "2026-05-01T10:00:05.000Z", parent="u-p1", inp=1, out=1),
        _user("p2", "2026-05-20T10:00:00.000Z", "late prompt", parent="u-r1"),
        _assistant("r2", "2026-05-20T10:00:05.000Z", parent="u-p2", inp=2, out=2),
    ]
    fake_claude.write("straddle.jsonl", events)

    run_extract(fake_claude.out, since="2026-05-10", timezone_name="UTC")

    prompts = _read_csv(fake_claude.out / "prompts.csv")
    assert [r["prompt_id"] for r in prompts] == ["p2"]
    # prompt_index reflects the position in the full session, not the window.
    assert prompts[0]["prompt_index"] == "2"
    assert len(_read_csv(fake_claude.out / "sessions.csv")) == 1


def test_until_bound_includes_last_millisecond(fake_claude):
    events = [
        _user("p1", "2026-05-31T23:59:59.500Z", "last-moment prompt"),
        _assistant("r1", "2026-05-31T23:59:59.900Z", parent="u-p1", inp=1, out=1),
    ]
    fake_claude.write("midnight.jsonl", events)

    run_extract(fake_claude.out, until="2026-05-31", timezone_name="UTC")
    assert [r["prompt_id"] for r in _read_csv(fake_claude.out / "prompts.csv")] == ["p1"]


def test_timezone_shifts_the_window(fake_claude):
    events = [
        _user("p1", "2026-05-31T23:30:00.000Z", "late UTC prompt"),
        _assistant("r1", "2026-05-31T23:30:05.000Z", parent="u-p1", inp=1, out=1),
    ]
    fake_claude.write("tz.jsonl", events)

    run_extract(fake_claude.out, until="2026-05-31", timezone_name="UTC")
    assert len(_read_csv(fake_claude.out / "prompts.csv")) == 1

    # In Tokyo (UTC+9) the prompt happened on June 1st.
    run_extract(fake_claude.out, until="2026-05-31", timezone_name="Asia/Tokyo")
    assert len(_read_csv(fake_claude.out / "prompts.csv")) == 0


def test_invalid_timezone_or_date_raises(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    with pytest.raises(ValueError, match="timezone"):
        run_extract(fake_claude.out, timezone_name="Mars/Olympus")
    with pytest.raises(ValueError, match="since"):
        run_extract(fake_claude.out, since="31/05/2026")


# ---------------------------------------------------------------------------
# Input robustness (3.10).
# ---------------------------------------------------------------------------


def test_bom_file_is_parsed_from_first_line(fake_claude):
    events = [
        _user("pB", "2026-06-01T10:00:00.000Z", "prompt behind a BOM"),
        _assistant("rB", "2026-06-01T10:00:05.000Z", parent="u-pB", inp=3, out=4),
    ]
    fake_claude.write("bom.jsonl", events, encoding="utf-8-sig")

    report = run_extract(fake_claude.out)
    # The first line (the prompt!) is not lost to the BOM.
    assert report.prompts == 1
    assert report.lines_invalid == 0


def test_corrupt_binary_file_is_skipped_not_fatal(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    bad = fake_claude.projects / "alpha" / "corrupt.jsonl"
    bad.write_bytes(b"\xff\xfe\x00garbage\x00binary")

    report = run_extract(fake_claude.out)
    assert report.prompts == 2  # alpha still extracted
    assert len(report.files_skipped) == 1
    assert "corrupt.jsonl" in report.files_skipped[0][0]
    assert report.exit_code() == 0
    assert report.exit_code(strict=True) == 1  # skipped file is a warning


def test_unreadable_path_is_skipped_not_fatal(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    # A directory matching *.jsonl: open() raises OSError, like a locked file.
    (fake_claude.projects / "alpha" / "locked.jsonl").mkdir()

    report = run_extract(fake_claude.out)
    assert report.prompts == 2
    assert len(report.files_skipped) == 1


def test_empty_and_invalid_json_lines(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    (fake_claude.projects / "alpha" / "empty.jsonl").write_text("", encoding="utf-8")
    (fake_claude.projects / "alpha" / "junk.jsonl").write_text(
        'not json at all\n{"type":"user"}\n', encoding="utf-8"
    )

    report = run_extract(fake_claude.out)
    assert report.prompts == 2
    assert report.files_read == 3
    assert report.lines_invalid == 1


# ---------------------------------------------------------------------------
# Extraction report (3.16 / 3.17).
# ---------------------------------------------------------------------------


def test_unpriced_model_is_reported(fake_claude):
    events = [
        _user("pM", "2026-06-01T10:00:00.000Z", "who are you"),
        _assistant(
            "rM", "2026-06-01T10:00:05.000Z", parent="u-pM", model="mystery-model-9", inp=5, out=5
        ),
    ]
    fake_claude.write("mystery.jsonl", events)

    report = run_extract(fake_claude.out)
    assert report.unpriced_models == ["mystery-model-9"]
    assert any("mystery-model-9" in w for w in report.warnings)
    assert report.exit_code() == 0
    assert report.exit_code(strict=True) == 1


def test_unknown_event_type_is_reported(fake_claude):
    fake_claude.add("session_delta.jsonl", project="delta")
    report = run_extract(fake_claude.out)
    assert report.unknown_event_types == {"flux-capacitor": 1}
    assert any("flux-capacitor" in w for w in report.warnings)


def test_zero_prompts_canary_fails_loudly(fake_claude):
    events = [_assistant("rZ", "2026-06-01T10:00:00.000Z", inp=5, out=5)]
    fake_claude.write("noprompts.jsonl", events)

    report = run_extract(fake_claude.out)
    assert report.files_read == 1
    assert report.prompts_total == 0
    assert report.exit_code() == 1
    assert any("0 prompts" in w for w in report.warnings)


def test_report_format_lines_mention_key_numbers(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    report = run_extract(fake_claude.out)
    text = "\n".join(report.format_lines())
    assert "Files read:      1" in text
    assert "Prompts:         2" in text
    assert "Request rows:    3" in text
    assert "duplicate(s) removed" in text
    assert "claude-opus-4-8" in text
    assert "versions 2.1.0" in text


# ---------------------------------------------------------------------------
# Output behaviour.
# ---------------------------------------------------------------------------


def test_no_text_omits_and_removes_text_file(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")

    run_extract(fake_claude.out)
    assert (fake_claude.out / "prompts_text.csv").exists()

    # With text, the preview column is populated.
    with_text = _read_csv(fake_claude.out / "prompts.csv")
    assert any(r["prompt_preview"] for r in with_text)

    run_extract(fake_claude.out, no_text=True)
    # A stale text file from a previous run is removed, not left behind.
    assert not (fake_claude.out / "prompts_text.csv").exists()
    # --no-text is honest: prompt_preview is blanked out too (10.1).
    no_text = _read_csv(fake_claude.out / "prompts.csv")
    assert no_text  # same prompts are still emitted
    assert all(r["prompt_preview"] == "" for r in no_text)


def test_token_types_meta_uses_machine_keys(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out)
    rows = _read_csv(fake_claude.out / "token_types.csv")
    keys = [r["token_type"] for r in rows]
    assert keys == [
        "input",
        "output",
        "cache_read",
        "cache_write_5m",
        "cache_write_1h",
        "server_tool_use",
    ]
    labels = {r["token_type"]: r["label"] for r in rows}
    assert labels["cache_write_1h"] == "Cache write (1h)"


def test_extract_never_touches_categories_csv(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    categories = fake_claude.out
    categories.mkdir(parents=True, exist_ok=True)
    sentinel = categories / "categories.csv"
    sentinel.write_text(
        "prompt_id,category,complexity,classifier_model,classified_at\n"
        "pA1,debug,2,test-model,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )

    run_extract(fake_claude.out)
    rows = _read_csv(sentinel)
    assert rows[0]["category"] == "debug"
    # And prompts.csv no longer carries category columns at all.
    prompts = _read_csv(fake_claude.out / "prompts.csv")
    assert "category" not in prompts[0]


# ---------------------------------------------------------------------------
# Axe C: output composition metrics (output_files.csv + output_tokens.csv).
# ---------------------------------------------------------------------------


def _output_files_by_prompt(out):
    """{prompt_id: {(language, kind): row}} from output_files.csv."""
    result: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for row in _read_csv(out / "output_files.csv"):
        result.setdefault(row["prompt_id"], {})[(row["language"], row["kind"])] = row
    return result


def test_output_files_language_kind_and_lines(fake_claude):
    fake_claude.add("session_output.jsonl", project="out")
    run_extract(fake_claude.out)

    files = _output_files_by_prompt(fake_claude.out)
    # pO1 writes then edits src/parser.py (one distinct Python code file).
    po1 = files["pO1"][("Python", "code")]
    assert po1["files"] == "1"
    assert po1["lines_added"] == "7"  # Write 5 + Edit +2
    assert po1["lines_deleted"] == "1"  # Edit -1
    # pO2 writes tests/test_parser.py -> Python test.
    po2 = files["pO2"][("Python", "test")]
    assert po2["files"] == "1"
    assert po2["lines_added"] == "3"
    assert po2["lines_deleted"] == "0"
    assert ("Python", "code") not in files["pO2"]


def test_output_tokens_split_reconciles_with_total_output(fake_claude):
    """prose + code per prompt equals that prompt's total output tokens."""
    fake_claude.add("session_output.jsonl", project="out")
    run_extract(fake_claude.out)

    tokens = _tokens_by_prompt(fake_claude.out)
    split = {
        r["prompt_id"]: (int(r["output_prose_tokens"]), int(r["output_code_tokens"]))
        for r in _read_csv(fake_claude.out / "output_tokens.csv")
    }
    for pid, (prose, code) in split.items():
        assert prose + code == tokens[pid]["output"], pid
        # Both blocks present in every assistant turn -> both sides non-zero.
        assert prose > 0 and code > 0, pid
    # pO1 = 50 + 40, pO2 = 60.
    assert sum(split["pO1"]) == 90
    assert sum(split["pO2"]) == 60


def test_output_split_all_prose_when_no_tool_blocks(fake_claude):
    """A pure-text answer (no tool_use) attributes all output to prose."""
    fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out)

    split = {
        r["prompt_id"]: (int(r["output_prose_tokens"]), int(r["output_code_tokens"]))
        for r in _read_csv(fake_claude.out / "output_tokens.csv")
    }
    tokens = _tokens_by_prompt(fake_claude.out)
    # session_alpha assistant lines carry no content blocks -> all prose.
    assert split["pA1"] == (tokens["pA1"]["output"], 0)
    assert split["pA2"] == (tokens["pA2"]["output"], 0)


def test_output_csvs_carry_metrics_only_no_source_code(fake_claude):
    """No file contents / edit strings / absolute paths ever reach the CSVs."""
    fake_claude.add("session_output.jsonl", project="out")
    run_extract(fake_claude.out)

    files_text = (fake_claude.out / "output_files.csv").read_text(encoding="utf-8")
    tokens_text = (fake_claude.out / "output_tokens.csv").read_text(encoding="utf-8")
    blob = files_text + tokens_text
    # Source fragments from the fixture must be absent.
    for secret in ("def parse", "return int", "import mod", "assert parse"):
        assert secret not in blob
    # No path column at all in output_files.csv, and no leaked path string.
    header = files_text.splitlines()[0]
    assert header == "prompt_id,language,kind,files,lines_added,lines_deleted"
    assert "parser.py" not in blob
    assert "/home/fake" not in blob


def test_output_files_dedup_across_resumed_replay(fake_claude):
    """A replayed tool call is not double-counted (dedup by tool_use id)."""
    fake_claude.add("session_output.jsonl", project="out")
    # A second copy of the same session: resumed sessions replay identical
    # lines; the edits must still count once.
    fake_claude.add("session_output.jsonl", project="out2")
    run_extract(fake_claude.out)

    files = _output_files_by_prompt(fake_claude.out)
    po1 = files["pO1"][("Python", "code")]
    assert po1["lines_added"] == "7"  # not 14
    assert po1["files"] == "1"


def test_output_csvs_respect_date_window(fake_claude):
    fake_claude.add("session_output.jsonl", project="out")
    run_extract(fake_claude.out, since="2026-06-05", timezone_name="UTC")
    # Both prompts predate the window -> empty (header-only) output CSVs.
    assert _read_csv(fake_claude.out / "output_files.csv") == []
    assert _read_csv(fake_claude.out / "output_tokens.csv") == []


# ---------------------------------------------------------------------------
# Axe D: context composition snapshot (context_sources.csv).
# ---------------------------------------------------------------------------

_CTX_CWD = "/home/fake/projects/ctx-proj"
_PY = "def parse(x):\n    return int(x)\n# a measured comment\n# another line\n"
_BASH_OUT = "tests/test_parser.py ... ok\n3 passed in 0.12s\n"
_GREP_OUT = "src/parser.py:1:def parse(x):\nsrc/util.py:4:def helper():\n"
_SKILL = "- verify: run the app and observe.\n- code-review: review the diff.\n"
_TS = "export function add(a: number, b: number): number {\n  return a + b\n}\n"

# The assistant turns, kept as variables so the test can recompute the exact
# conversation token weight (prose + code) the extractor will derive.
_A_READ = [
    {"type": "text", "text": "Reading the parser."},
    {
        "type": "tool_use",
        "id": "tR1",
        "name": "Read",
        "input": {"file_path": f"{_CTX_CWD}/src/parser.py"},
    },
]
_A_BASH = [
    {"type": "text", "text": "Running the suite."},
    {"type": "tool_use", "id": "tB1", "name": "Bash", "input": {"command": "pytest -q"}},
]
_A_GREP = [
    {"type": "text", "text": "Searching for definitions."},
    {"type": "tool_use", "id": "tG1", "name": "Grep", "input": {"pattern": "def "}},
]
_A_DONE = [{"type": "text", "text": "All done, the parser is fixed."}]
_PROMPT_TEXT = "Analyze the parser, run the tests, and search for definitions."


def _ctx_events(sid="sess-ctx"):
    """A session exercising every context source (read / output / config / chat)."""

    def u(uuid, parent, content, **extra):
        return {
            "type": "user",
            "uuid": uuid,
            "parentUuid": parent,
            "timestamp": "2026-06-08T10:00:00.000Z",
            "sessionId": sid,
            "cwd": _CTX_CWD,
            "gitBranch": "main",
            "message": {"role": "user", "content": content},
            **extra,
        }

    def a(uuid, parent, req, content):
        return {
            "type": "assistant",
            "uuid": uuid,
            "parentUuid": parent,
            "requestId": req,
            "timestamp": "2026-06-08T10:00:01.000Z",
            "sessionId": sid,
            "cwd": _CTX_CWD,
            "message": {
                "id": f"m-{req}",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "stop_reason": "end_turn",
                "content": content,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        }

    def result(tool_id, content):
        return [{"type": "tool_result", "tool_use_id": tool_id, "content": content}]

    def att(uuid, parent, attachment):
        return {
            "type": "attachment",
            "uuid": uuid,
            "parentUuid": parent,
            "timestamp": "2026-06-08T10:00:02.000Z",
            "sessionId": sid,
            "cwd": _CTX_CWD,
            "attachment": attachment,
        }

    return [
        u("uc1", None, _PROMPT_TEXT, promptId="pC1"),
        a("uc2", "uc1", "rc1", _A_READ),
        u("uc3", "uc2", result("tR1", _PY)),
        a("uc4", "uc3", "rc2", _A_BASH),
        u("uc5", "uc4", result("tB1", _BASH_OUT)),
        a("uc6", "uc5", "rc3", _A_GREP),
        # tool_result body as a list of text blocks (the other valid shape).
        u("uc7", "uc6", result("tG1", [{"type": "text", "text": _GREP_OUT}])),
        att("uc8", "uc7", {"type": "skill_listing", "content": _SKILL}),
        att("uc9", "uc8", {"type": "file", "filename": f"{_CTX_CWD}/web/app.ts", "content": _TS}),
        a("uc10", "uc9", "rc4", _A_DONE),
    ]


def _context_by_source(out):
    """{(source, language): {'tokens': int, 'items': int}} from context_sources.csv."""
    result: dict[tuple[str, str], dict[str, int]] = {}
    for row in _read_csv(out / "context_sources.csv"):
        bucket = result.setdefault((row["source"], row["language"]), {"tokens": 0, "items": 0})
        bucket["tokens"] += int(row["tokens"])
        bucket["items"] += int(row["items"])
    return result


def test_context_sources_sizes_each_source(fake_claude):
    """Each source is sized by the local tokenizer; file reads keep a language."""
    fake_claude.write("session_context.jsonl", _ctx_events(), project="ctx")
    run_extract(fake_claude.out)

    by_source = _context_by_source(fake_claude.out)

    # Files read: the Read result (Python) and the file attachment (TypeScript).
    assert by_source[("file_read", "Python")] == {"tokens": count_tokens(_PY), "items": 1}
    assert by_source[("file_read", "TypeScript")] == {"tokens": count_tokens(_TS), "items": 1}
    # Tool output: Bash + Grep results, both language-less.
    assert by_source[("tool_output", "-")] == {
        "tokens": count_tokens(_BASH_OUT) + count_tokens(_GREP_OUT),
        "items": 2,
    }
    # Config: the injected skill listing (no filename -> config bucket).
    assert by_source[("config", "-")] == {"tokens": count_tokens(_SKILL), "items": 1}
    # Conversation: the prompt + every main-thread assistant turn (prose + code).
    expected_conv = count_tokens(_PROMPT_TEXT)
    for content in (_A_READ, _A_BASH, _A_GREP, _A_DONE):
        prose, code, _ = analyze_assistant_content(content, _CTX_CWD)
        expected_conv += prose + code
    assert by_source[("conversation", "-")] == {"tokens": expected_conv, "items": 5}


def test_context_source_shares_sum_to_one(fake_claude):
    """The per-source token shares are a partition of the total context size."""
    fake_claude.write("session_context.jsonl", _ctx_events(), project="ctx")
    run_extract(fake_claude.out)

    rows = _read_csv(fake_claude.out / "context_sources.csv")
    total = sum(int(r["tokens"]) for r in rows)
    assert total > 0
    assert {r["source"] for r in rows} == {"conversation", "file_read", "tool_output", "config"}


def test_context_sources_metrics_only_no_content(fake_claude):
    """No file content / tool output / paths ever reach context_sources.csv."""
    fake_claude.write("session_context.jsonl", _ctx_events(), project="ctx")
    run_extract(fake_claude.out)

    blob = (fake_claude.out / "context_sources.csv").read_text(encoding="utf-8")
    header = blob.splitlines()[0]
    assert header == "session_id,source,language,tokens,items"
    for secret in ("def parse", "return int", "3 passed", "def helper", "verify", "app.ts"):
        assert secret not in blob
    assert "/home/fake" not in blob and "parser.py" not in blob


def test_context_sources_dedup_across_resumed_replay(fake_claude):
    """A replayed session does not double-count reads / output / config."""
    fake_claude.write("session_context.jsonl", _ctx_events(), project="ctx")
    run_extract(fake_claude.out)
    once = _context_by_source(fake_claude.out)

    # A resumed copy replays identical uuids / tool_use ids: totals must hold.
    fake_claude.write("session_context_resumed.jsonl", _ctx_events(), project="ctx2")
    run_extract(fake_claude.out)
    twice = _context_by_source(fake_claude.out)
    assert twice == once


def test_context_sources_respect_date_window(fake_claude):
    """Out-of-window sessions drop from the context snapshot entirely."""
    fake_claude.write("session_context.jsonl", _ctx_events(), project="ctx")
    run_extract(fake_claude.out, since="2026-07-01", timezone_name="UTC")
    assert _read_csv(fake_claude.out / "context_sources.csv") == []


# ---------------------------------------------------------------------------
# Axe D (D2): context cost over time (context_cost.csv) -- the rigour signature
# is that the attributed cache tokens reconcile to the billed main chain.
# ---------------------------------------------------------------------------


def _cost_events(sid="sess-cost"):
    """A multi-turn session carrying real cache usage across its requests."""

    def a(uuid, parent, req, content, usage):
        return {
            "type": "assistant",
            "uuid": uuid,
            "parentUuid": parent,
            "requestId": req,
            "timestamp": f"2026-06-08T10:0{req[-1]}:01.000Z",
            "sessionId": sid,
            "cwd": _CTX_CWD,
            "message": {
                "id": f"m-{req}",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "stop_reason": "end_turn",
                "content": content,
                "usage": usage,
            },
        }

    def u(uuid, parent, content, **extra):
        return {
            "type": "user",
            "uuid": uuid,
            "parentUuid": parent,
            "timestamp": "2026-06-08T10:00:00.000Z",
            "sessionId": sid,
            "cwd": _CTX_CWD,
            "message": {"role": "user", "content": content},
            **extra,
        }

    def result(tool_id, content):
        return [{"type": "tool_result", "tool_use_id": tool_id, "content": content}]

    return [
        u("u1", None, _PROMPT_TEXT, promptId="pc1"),
        # Turn 1: fresh input + big cache writes (loading the context).
        a(
            "a1",
            "u1",
            "req1",
            _A_READ,
            {
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_creation_input_tokens": 8000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 6000,
                    "ephemeral_1h_input_tokens": 2000,
                },
            },
        ),
        u("u2", "a1", result("tR1", _PY)),
        # Turn 2: the context is hot, re-read as cache_read (rent).
        a(
            "a2",
            "u2",
            "req2",
            _A_BASH,
            {
                "input_tokens": 5,
                "output_tokens": 15,
                "cache_read_input_tokens": 9000,
                "cache_creation_input_tokens": 500,
                "cache_creation": {"ephemeral_5m_input_tokens": 500},
            },
        ),
        u("u3", "a2", result("tB1", _BASH_OUT)),
        a(
            "a3",
            "u3",
            "req3",
            _A_DONE,
            {"input_tokens": 5, "output_tokens": 30, "cache_read_input_tokens": 12000},
        ),
    ]


def _context_cost_rows(out):
    return _read_csv(out / "context_cost.csv")


def test_context_cost_reconciles_with_the_billed_main_chain(fake_claude):
    """Attributed rent/load equal the billed main-chain cache tokens, to the token."""
    fake_claude.write("session_cost.jsonl", _cost_events(), project="cost")
    run_extract(fake_claude.out)

    cost = _context_cost_rows(fake_claude.out)
    requests = _read_csv(fake_claude.out / "requests.csv")
    main = [r for r in requests if r["is_sidechain"] == "0"]

    assert sum(int(r["rent_read_tokens"]) for r in cost) == sum(
        int(r["cache_read_tokens"]) for r in main
    )
    assert sum(int(r["load_write_5m_tokens"]) for r in cost) == sum(
        int(r["cache_write_5m_tokens"]) for r in main
    )
    assert sum(int(r["load_write_1h_tokens"]) for r in cost) == sum(
        int(r["cache_write_1h_tokens"]) for r in main
    )
    # The session loaded the context once (writes) then paid rent re-reading it.
    assert sum(int(r["rent_read_tokens"]) for r in cost) == 21000
    assert sum(int(r["load_write_5m_tokens"]) for r in cost) == 6500


def test_context_cost_metrics_only_no_content(fake_claude):
    """context_cost.csv carries raw token counts only -- no content, no paths."""
    fake_claude.write("session_cost.jsonl", _cost_events(), project="cost")
    run_extract(fake_claude.out)

    blob = (fake_claude.out / "context_cost.csv").read_text(encoding="utf-8")
    assert blob.splitlines()[0] == (
        "session_id,source,language,model,rent_read_tokens,"
        "load_write_5m_tokens,load_write_1h_tokens"
    )
    for secret in ("def parse", "return int", "3 passed", "parser.py"):
        assert secret not in blob
    assert "/home/fake" not in blob


def test_context_cost_dedup_across_resumed_replay(fake_claude):
    """A replayed session does not double-count the attributed cache cost."""
    fake_claude.write("session_cost.jsonl", _cost_events(), project="cost")
    run_extract(fake_claude.out)
    once = sum(int(r["rent_read_tokens"]) for r in _context_cost_rows(fake_claude.out))

    fake_claude.write("session_cost_resumed.jsonl", _cost_events(), project="cost2")
    run_extract(fake_claude.out)
    twice = sum(int(r["rent_read_tokens"]) for r in _context_cost_rows(fake_claude.out))
    assert twice == once


def test_context_cost_respects_date_window(fake_claude):
    """Out-of-window sessions drop from the context cost entirely."""
    fake_claude.write("session_cost.jsonl", _cost_events(), project="cost")
    run_extract(fake_claude.out, since="2026-07-01", timezone_name="UTC")
    assert _context_cost_rows(fake_claude.out) == []
