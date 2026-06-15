"""Tests for the request-grain analyses (phase 2: 2.1-2.4).

Every aggregation is verified against hand-computed values on a small
synthetic dataset. Pricing used (bundled pricing.yml, per 1M tokens):
claude-opus-4-8 / anthropic: input 5, output 25, cache_read 0.5,
cache_write_5m 6.25, cache_write_1h 10.

The synthetic session s1 tells one story end to end:

* p1 (depth 1, 3 requests): cold start writing 18k of context, then two
  cheap incremental turns 60s apart (gaps in the <= 5m baseline bucket).
* p2 (depth 2, 1 request): resumes after a 2h03m pause -- every cache entry
  expired, the request re-writes the full 22k context (the TTL spike).
* p3 (depth 3, 1 request) then a compaction: the synthetic continuation
  (s1:_continuation, post_compact=1) restarts from an 11k summary context.
* One sidechain request inside p1 must be excluded from every analysis.

s2 is a one-prompt session (10k first-turn context) so that medians and the
overhead analysis aggregate across sessions.
"""

from __future__ import annotations

import pytest

from prompt_analytics import analytics
from prompt_analytics.analytics import Dataset
from prompt_analytics.extract import run_extract

OPUS = "claude-opus-4-8"


def _request(
    session_id,
    prompt_id,
    request_index,
    timestamp,
    *,
    input=0,
    output=0,
    cache_read=0,
    cw_5m=0,
    cw_1h=0,
    side=0,
    post_compact=0,
    model=OPUS,
):
    return {
        "session_id": session_id,
        "prompt_id": prompt_id,
        "request_index": request_index,
        "timestamp": timestamp,
        "model": model,
        "stop_reason": "end_turn",
        "is_sidechain": side,
        "post_compact": post_compact,
        "input_tokens": input,
        "output_tokens": output,
        "cache_read_tokens": cache_read,
        "cache_write_5m_tokens": cw_5m,
        "cache_write_1h_tokens": cw_1h,
        "server_tool_use_requests": 0,
    }


def _prompt(session_id, prompt_id, index, timestamp, project):
    return {
        "session_id": session_id,
        "prompt_id": prompt_id,
        "prompt_index": index,
        "timestamp": timestamp,
        "project": project,
        "model": OPUS,
        "assistant_turns": 1,
        "tool_calls": 0,
    }


@pytest.fixture
def ds() -> Dataset:
    """Hand-computed request-grain dataset (see the module docstring).

    Per-request anthropic costs (context = input + cache_read + writes):

    * r1: 2k in ($0.01) + 500 out ($0.0125) + 18k cw1h ($0.18) = $0.2025,
      context 20,000.
    * r2: 100 in + 300 out + 20k read + 1k cw1h = $0.028, context 21,100.
    * r3: 50 in + 200 out + 21k read + 950 cw1h = $0.02525, context 22,000.
    * r4 (after 2h03m): 200 in + 600 out + 22k cw1h = $0.236, context 22,200.
    * r5 (after 10m): 100 in + 300 out + 22k read + 2k cw1h = $0.039,
      context 24,100.
    * r6 (continuation, post-compact, after 5m): 5k in + 800 out + 6k cw1h,
      context 11,000, rebuild write 6,000 ($0.06).
    * r7 (post-compact, after 60s): 50 in + 100 out + 11k read + 500 cw1h,
      writes 500.
    * r8 (s2 opener): 1k in + 400 out + 9k cw1h = $0.105, context 10,000.
    * r9: sidechain inside p1, excluded everywhere.
    """
    sessions = [
        {"session_id": "s1", "start_date": "2026-05-01", "project": "alpha"},
        {"session_id": "s2", "start_date": "2026-05-02", "project": "beta"},
    ]
    prompts = [
        _prompt("s1", "p1", 1, "2026-05-01T10:00:00Z", "alpha"),
        _prompt("s1", "p2", 2, "2026-05-01T12:05:00Z", "alpha"),
        _prompt("s1", "p3", 3, "2026-05-01T12:15:00Z", "alpha"),
        _prompt("s2", "q1", 1, "2026-05-02T09:00:00Z", "beta"),
    ]
    requests = [
        _request("s1", "p1", 1, "2026-05-01T10:00:00Z", input=2_000, output=500, cw_1h=18_000),
        _request(
            "s1",
            "p1",
            2,
            "2026-05-01T10:01:00Z",
            input=100,
            output=300,
            cache_read=20_000,
            cw_1h=1_000,
        ),
        _request("s1", "p1", 3, "2026-05-01T10:01:30Z", input=10_000, output=2_000, side=1),
        _request(
            "s1",
            "p1",
            4,
            "2026-05-01T10:02:00Z",
            input=50,
            output=200,
            cache_read=21_000,
            cw_1h=950,
        ),
        _request("s1", "p2", 1, "2026-05-01T12:05:00Z", input=200, output=600, cw_1h=22_000),
        _request(
            "s1",
            "p3",
            1,
            "2026-05-01T12:15:00Z",
            input=100,
            output=300,
            cache_read=22_000,
            cw_1h=2_000,
        ),
        _request(
            "s1",
            "s1:_continuation",
            1,
            "2026-05-01T12:20:00Z",
            input=5_000,
            output=800,
            cw_1h=6_000,
            post_compact=1,
        ),
        _request(
            "s1",
            "s1:_continuation",
            2,
            "2026-05-01T12:21:00Z",
            input=50,
            output=100,
            cache_read=11_000,
            cw_1h=500,
            post_compact=1,
        ),
        _request("s2", "q1", 1, "2026-05-02T09:00:00Z", input=1_000, output=400, cw_1h=9_000),
    ]
    return Dataset(
        sessions=sessions,
        prompts=prompts,
        tokens=[],
        categories={},
        source="test data",
        requests=requests,
    )


def _rows_by(result, key):
    return {row[key]: row for row in result.rows}


# ---------------------------------------------------------------------------
# 2.1 Accumulated context by depth.
# ---------------------------------------------------------------------------


def test_context_growth_hand_computed(ds):
    result = analytics.context_growth(ds, "anthropic")
    by_depth = _rows_by(result, "depth")

    # Depth 1 = r1, r2, r4(main r3) of p1 + r8 of q1; the sidechain request is
    # excluded, the continuation has no prompt_index.
    d1 = by_depth["1"]
    assert d1["requests"] == 4
    # contexts [20000, 21100, 22000, 10000] -> median 20550, p90 = 22000.
    assert d1["median_context"] == 20_550
    assert d1["p90_context"] == 22_000
    assert d1["cache_read_per_turn"] == 41_000 // 4
    assert d1["vs_depth_1"] == 1.0
    assert d1["cost_per_request_usd"] == round(0.36075 / 4, 4)

    d2 = by_depth["2"]
    assert d2["requests"] == 1
    assert d2["median_context"] == 22_200
    assert d2["vs_depth_1"] == 1.08  # 22200 / 20550
    assert d2["cost_per_request_usd"] == 0.236

    d3 = by_depth["3"]
    assert d3["median_context"] == 24_100
    assert d3["cache_read_per_turn"] == 22_000
    assert d3["vs_depth_1"] == 1.17  # 24100 / 20550

    assert len(result.rows) == 3
    assert any("Median context at depth 3" in note for note in result.notes)


def test_context_growth_without_requests_says_so():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test")
    result = analytics.context_growth(empty, "anthropic")
    assert result.rows == []
    assert any("requests.csv" in note for note in result.notes)


# ---------------------------------------------------------------------------
# 2.2 TTL expiry losses.
# ---------------------------------------------------------------------------


def test_ttl_losses_hand_computed(ds):
    result = analytics.ttl_losses(ds, "anthropic")
    by_gap = _rows_by(result, "gap")

    # Baseline gaps (60s, 60s in p1; 60s before r7): writes 1000, 950, 500
    # -> median 950, avg 816.
    base = by_gap["<= 5m"]
    assert base["events"] == 3
    assert base["avg_write_tokens"] == 2_450 // 3
    assert base["write_cost_usd"] == round(2_450 * 10 / 1e6, 4)
    assert base["excess_cost_usd"] is None

    # 10m gap before r5 (writes 2000, $0.02) and 5m gap before r6 (writes
    # 6000, $0.06): excess over the 950 baseline, prorated by cost.
    mid = by_gap["5m-1h"]
    assert mid["events"] == 2
    assert mid["avg_write_tokens"] == 4_000
    assert mid["write_cost_usd"] == 0.08
    assert mid["excess_cost_usd"] == round(0.02 * 1_050 / 2_000 + 0.06 * 5_050 / 6_000, 4)

    # The 2h03m pause before r4: the full 22k re-write is the spike.
    long = by_gap["1h-6h"]
    assert long["events"] == 1
    assert long["avg_write_tokens"] == 22_000
    assert long["write_cost_usd"] == 0.22
    assert long["excess_cost_usd"] == round(0.22 * 21_050 / 22_000, 4)

    assert "> 6h" not in by_gap  # no such pause in the data
    assert any("median 950" in note for note in result.notes)
    assert any("$0.21" in note and "pauses > 1h" in note for note in result.notes)


def test_ttl_losses_without_requests_says_so():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test")
    result = analytics.ttl_losses(empty, "anthropic")
    assert result.rows == []
    assert any("requests.csv" in note for note in result.notes)


# ---------------------------------------------------------------------------
# 2.3 Compactions.
# ---------------------------------------------------------------------------


def test_compactions_hand_computed(ds):
    result = analytics.compactions(ds, "anthropic")
    assert len(result.rows) == 1
    event = result.rows[0]
    assert event["session_id"] == "s1"
    assert event["project"] == "alpha"
    assert event["context_before"] == 24_100  # r5, the last pre-compact request
    assert event["context_after"] == 11_000  # r6, restarted from the summary
    assert event["reduction_pct"] == round(100 * (1 - 11_000 / 24_100), 1)
    assert event["rebuild_tokens"] == 6_000
    assert event["rebuild_cost_usd"] == 0.06
    # r7 continues the same post-compact run: no second event.
    assert any("1 compaction(s) across 1 of 2 sessions" in note for note in result.notes)
    assert any("$0.06" in note for note in result.notes)


def test_compactions_none_found(ds):
    ds.requests = [row for row in ds.requests if not row["post_compact"]]
    result = analytics.compactions(ds, "anthropic")
    assert result.rows == []
    assert any("No compaction found" in note for note in result.notes)


def test_compactions_on_delta_fixture(fake_claude):
    """Hand-verified against session_delta.jsonl: rD4 answers the compact
    summary; the request before it is rD2 (in 20 + read 400 = context 420),
    rD4 itself is in 5 + cw5m 1000 = context 1005, rebuild 1000 tokens."""
    fake_claude.add("session_delta.jsonl", project="delta")
    run_extract(fake_claude.out)
    ds = analytics.dataset_from_csvs(fake_claude.out)

    result = analytics.compactions(ds, "anthropic")
    assert len(result.rows) == 1
    event = result.rows[0]
    assert event["session_id"] == "sess-ddd"
    assert event["context_before"] == 420
    assert event["context_after"] == 1_005
    assert event["rebuild_tokens"] == 1_000
    assert event["rebuild_cost_usd"] == round(1_000 * 6.25 / 1e6, 4)


# ---------------------------------------------------------------------------
# 2.4 Fixed session overhead.
# ---------------------------------------------------------------------------


def test_session_overhead_hand_computed(ds):
    result = analytics.session_overhead(ds, "anthropic")
    assert [row["session_id"] for row in result.rows] == ["s1", "s2"]  # by date
    s1, s2 = result.rows
    assert s1["overhead_tokens"] == 20_000  # r1: 2k input + 18k cache write
    assert s1["cache_write_tokens"] == 18_000
    assert s1["first_request_cost_usd"] == 0.2025
    assert s1["started"] == "2026-05-01"
    assert s2["overhead_tokens"] == 10_000
    assert s2["first_request_cost_usd"] == 0.105
    # Median across the two sessions: (20000 + 10000) / 2.
    assert any("15,000 tokens" in note for note in result.notes)
    assert any("$0.31 in total" in note for note in result.notes)


def test_session_overhead_without_requests_says_so():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test")
    result = analytics.session_overhead(empty, "anthropic")
    assert result.rows == []
    assert any("requests.csv" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Dataset loading: the request grain rides along (live, CSV, freshness).
# ---------------------------------------------------------------------------


def test_load_dataset_carries_requests_live_and_csv(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    live = analytics.load_dataset(fake_claude.out)
    assert "live parse" in live.source
    assert len(live.requests) == 3  # rA1a (deduplicated), rA1b, rA2a

    run_extract(fake_claude.out)
    cached = analytics.load_dataset(fake_claude.out)
    assert "(fresh)" in cached.source
    # Both paths agree, ints coerced on the CSV path.
    key = ("prompt_id", "request_index", "input_tokens", "cache_write_5m_tokens")
    assert [tuple(r[k] for k in key) for r in cached.requests] == [
        tuple(r[k] for k in key) for r in live.requests
    ]


def test_missing_requests_csv_is_never_served_as_fresh(fake_claude):
    """A pre-v2 extract (no requests.csv) must fall back to a live parse:
    the request-grain analyses would otherwise run on a silently empty table."""
    fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out)
    (fake_claude.out / "requests.csv").unlink()
    loaded = analytics.load_dataset(fake_claude.out)
    assert "live parse" in loaded.source
    assert loaded.requests  # rebuilt from the JSONL history
