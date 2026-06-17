"""Tests for the power-user "what to do" analyses (phase 2: 2.5-2.7).

Every aggregation is verified against hand-computed values. Pricing used
(bundled pricing.yml, per 1M tokens):

* claude-opus-4-8 : input 5, output 25, cache_read 0.5, cw_5m 6.25, cw_1h 10
* claude-sonnet-4-6: input 3, output 15, cache_read 0.3, cw_5m 3.75, cw_1h 6
"""

from __future__ import annotations

import pytest

from prompt_analytics import analytics
from prompt_analytics.analytics import Dataset

OPUS = "claude-opus-4-8"
SONNET = "claude-sonnet-4-6"


def _tok(session, pid, model, token_type, count, *, side=0):
    return {
        "session_id": session,
        "prompt_id": pid,
        "model": model,
        "token_type": token_type,
        "is_sidechain": side,
        "token_count": count,
    }


def _rows_by(result, key):
    return {row[key]: row for row in result.rows}


# ---------------------------------------------------------------------------
# 2.5 Model x category cross + what-if re-pricing.
# ---------------------------------------------------------------------------


@pytest.fixture
def cross_ds() -> Dataset:
    """Two opus 'code' prompts and one sonnet 'docs' prompt.

    * p1 (opus, code): input 1,000 ($0.005) + output 200 ($0.005)
      + cache_read 100,000 ($0.05) = $0.06, 101,200 tokens.
    * p2 (opus, code): cache_read 200,000 ($0.10), 200,000 tokens.
    * p3 (sonnet, docs): input 1,000 ($0.003) + output 500 ($0.0075)
      = $0.0105, 1,500 tokens.
    """
    prompts = [
        {"session_id": "s1", "prompt_id": "p1", "project": "alpha"},
        {"session_id": "s1", "prompt_id": "p2", "project": "alpha"},
        {"session_id": "s2", "prompt_id": "p3", "project": "beta"},
    ]
    tokens = [
        _tok("s1", "p1", OPUS, "input", 1_000),
        _tok("s1", "p1", OPUS, "output", 200),
        _tok("s1", "p1", OPUS, "cache_read", 100_000),
        _tok("s1", "p2", OPUS, "cache_read", 200_000),
        _tok("s2", "p3", SONNET, "input", 1_000),
        _tok("s2", "p3", SONNET, "output", 500),
    ]
    categories = {
        "p1": {"category": "code", "complexity": "3"},
        "p2": {"category": "code", "complexity": "4"},
        "p3": {"category": "docs", "complexity": "1"},
    }
    return Dataset(
        sessions=[],
        prompts=prompts,
        tokens=tokens,
        categories=categories,
        source="test data",
    )


def test_model_category_cross_hand_computed(cross_ds):
    result = analytics.model_category(cross_ds, "anthropic")
    rows = result.rows
    # Sorted by cost desc: (opus, code) then (sonnet, docs).
    assert (rows[0]["model"], rows[0]["category"]) == (OPUS, "code")
    assert rows[0]["prompts"] == 2
    assert rows[0]["tokens"] == 301_200
    assert rows[0]["cost_usd"] == 0.16  # 0.06 + 0.10
    assert rows[1]["model"] == SONNET
    assert rows[1]["cost_usd"] == 0.0105
    # No re-pricing columns without --whatif.
    assert "repriced_usd" not in rows[0]


def test_model_category_whatif_reprices_on_target(cross_ds):
    result = analytics.model_category(cross_ds, "anthropic", target_model=SONNET)
    by_model = _rows_by(result, "model")

    # opus/code on sonnet: input 1k*3 + output 200*15 + read 100k*0.3 = 0.036
    # (p1) ; read 200k*0.3 = 0.06 (p2) -> 0.096. Saving 0.16 - 0.096 = 0.064.
    opus = by_model[OPUS]
    assert opus["repriced_usd"] == 0.096
    assert opus["saving_usd"] == 0.064
    # sonnet/docs re-priced on sonnet: unchanged, zero saving.
    sonnet = by_model[SONNET]
    assert sonnet["repriced_usd"] == 0.0105
    assert sonnet["saving_usd"] == 0.0
    # Headline: total 0.1705 -> 0.1065 (save 0.064).
    assert any("$0.17" in note and "$0.11" in note for note in result.notes)
    assert SONNET in result.title


# ---------------------------------------------------------------------------
# 2.6 Compaction recommendations.
# ---------------------------------------------------------------------------


def _prompt(session, pid, index, project="alpha"):
    return {
        "session_id": session,
        "prompt_id": pid,
        "prompt_index": index,
        "timestamp": "",
        "project": project,
    }


def _request(session, pid, request_index, *, input=0, cache_read=0, cw_1h=0, model=OPUS):
    return {
        "session_id": session,
        "prompt_id": pid,
        "request_index": request_index,
        "timestamp": "",
        "model": model,
        "stop_reason": "end_turn",
        "is_sidechain": 0,
        "post_compact": 0,
        "input_tokens": input,
        "output_tokens": 0,
        "cache_read_tokens": cache_read,
        "cache_write_5m_tokens": 0,
        "cache_write_1h_tokens": cw_1h,
        "server_tool_use_requests": 0,
    }


@pytest.fixture
def long_ds() -> Dataset:
    """s1 has 3 main-chain prompts (depths 1-3); s2 is a one-prompt session.

    No compaction in the data, so the baseline falls back to the median
    first-turn context: s1 r1 = 5k in + 5k cw_1h = 10,000; s2 = 10k in + 10k
    cw_1h = 20,000 -> median baseline 15,000.

    Main-chain s1 cache rent:
    * r1 (depth 1): cw_1h 5,000 -> $0.05.
    * r2 (depth 2): cache_read 50,000 -> $0.025.
    * r3 (depth 3): cache_read 500,000 ($0.25) + cw_1h 1,000 ($0.01) = $0.26.
    rent = $0.335.

    With compact_at=2, only r3 (depth 3) is past the threshold: excess read
    500k - 15k = 485k -> saved $0.2425; rebuild 15k cw_1h -> $0.15;
    net saving $0.0925; est. if compacted = 0.335 - 0.0925 = $0.2425.
    """
    sessions = [
        {"session_id": "s1", "project": "alpha"},
        {"session_id": "s2", "project": "beta"},
    ]
    prompts = [
        _prompt("s1", "p1", 1),
        _prompt("s1", "p2", 2),
        _prompt("s1", "p3", 3),
        _prompt("s2", "q1", 1, project="beta"),
    ]
    requests = [
        _request("s1", "p1", 1, input=5_000, cw_1h=5_000),
        _request("s1", "p2", 1, cache_read=50_000),
        _request("s1", "p3", 1, cache_read=500_000, cw_1h=1_000),
        _request("s2", "q1", 1, input=10_000, cw_1h=10_000),
    ]
    return Dataset(
        sessions=sessions,
        prompts=prompts,
        tokens=[],
        categories={},
        source="test data",
        requests=requests,
    )


def test_recommendations_hand_computed(long_ds):
    result = analytics.recommendations(long_ds, "anthropic", min_prompts=2, compact_at=2)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["session_id"] == "s1"
    assert row["prompts"] == 3
    assert row["rent_usd"] == 0.335
    assert row["saving_usd"] == 0.0925
    assert row["est_compacted_usd"] == 0.2425
    assert any("paid $0.34 in cache rent" in note for note in result.notes)
    assert any("15,000 tokens" in note for note in result.notes)


def test_recommendations_no_long_sessions(long_ds):
    result = analytics.recommendations(long_ds, "anthropic")  # min_prompts=50
    assert result.rows == []
    assert any("No session exceeds 50 prompts" in note for note in result.notes)


def test_recommendations_without_requests_says_so():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test")
    result = analytics.recommendations(empty, "anthropic")
    assert result.rows == []
    assert any("requests.csv" in note for note in result.notes)


# ---------------------------------------------------------------------------
# 2.7 Weekly burn rate.
# ---------------------------------------------------------------------------


def _dated_prompt(pid, timestamp):
    return {"session_id": "s", "prompt_id": pid, "prompt_index": 1, "timestamp": timestamp}


@pytest.fixture
def burn_ds() -> Dataset:
    """Three opus prompts (output only, $0.025 per 1k):

    * 2026-05-04 (Mon): output 1,000 -> $0.025.
    * 2026-05-05 (Tue): output 2,000 -> $0.05. (week of 2026-05-04)
    * 2026-05-11 (Mon): output 4,000 -> $0.10. (week of 2026-05-11)
    """
    prompts = [
        _dated_prompt("p1", "2026-05-04T10:00:00Z"),
        _dated_prompt("p2", "2026-05-05T10:00:00Z"),
        _dated_prompt("p3", "2026-05-11T10:00:00Z"),
    ]
    tokens = [
        _tok("s", "p1", OPUS, "output", 1_000),
        _tok("s", "p2", OPUS, "output", 2_000),
        _tok("s", "p3", OPUS, "output", 4_000),
    ]
    return Dataset(sessions=[], prompts=prompts, tokens=tokens, categories={}, source="test data")


def test_burn_rate_hand_computed(burn_ds):
    result = analytics.burn_rate(burn_ds, "anthropic")
    by_week = _rows_by(result, "week_of")

    week_a = by_week["2026-05-04"]
    assert week_a["active_days"] == 2
    assert week_a["prompts"] == 2
    assert week_a["cost_usd"] == 0.075
    assert week_a["per_day_usd"] == round(0.075 / 7, 4)
    assert week_a["vs_prev_pct"] is None  # first week

    week_b = by_week["2026-05-11"]
    assert week_b["active_days"] == 1
    assert week_b["cost_usd"] == 0.1
    assert week_b["vs_prev_pct"] == 33.3  # (0.10 - 0.075) / 0.075

    # Span 2026-05-04 .. 2026-05-11 = 8 days, total $0.175.
    assert any("Burn rate" in note and "8-day span" in note for note in result.notes)
    # Last 7 days (05-05..05-11) = 0.15 vs prior 7 (04-28..05-04) = 0.025.
    assert any("Last 7 days $0.15" in note for note in result.notes)


def test_burn_rate_weeks_limit(burn_ds):
    result = analytics.burn_rate(burn_ds, "anthropic", weeks=1)
    assert len(result.rows) == 1
    assert result.rows[0]["week_of"] == "2026-05-11"  # most recent kept


def test_burn_rate_no_dated_prompts():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test")
    result = analytics.burn_rate(empty, "anthropic")
    assert result.rows == []
    assert any("No dated prompts" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Timeline: cost/prompts/tokens grouped by day / week / month.
# ---------------------------------------------------------------------------


def test_timeline_by_day(burn_ds):
    result = analytics.timeline(burn_ds, "anthropic", by="day")
    assert result.title == "Cost by day"
    assert [row["period"] for row in result.rows] == ["2026-05-04", "2026-05-05", "2026-05-11"]
    by_day = _rows_by(result, "period")
    assert by_day["2026-05-04"]["cost_usd"] == 0.025
    assert by_day["2026-05-04"]["tokens"] == 1_000
    assert by_day["2026-05-04"]["prompts"] == 1
    assert by_day["2026-05-11"]["cost_usd"] == 0.1
    # Shares are of the $0.175 total.
    assert by_day["2026-05-11"]["share_pct"] == 57.1


def test_timeline_by_week_matches_burn_rate_weeks(burn_ds):
    result = analytics.timeline(burn_ds, "anthropic", by="week")
    by_week = _rows_by(result, "period")
    assert by_week["2026-05-04"]["cost_usd"] == 0.075
    assert by_week["2026-05-11"]["cost_usd"] == 0.1


def test_timeline_by_month(burn_ds):
    result = analytics.timeline(burn_ds, "anthropic", by="month")
    assert [row["period"] for row in result.rows] == ["2026-05"]
    assert result.rows[0]["cost_usd"] == 0.175
    assert result.rows[0]["tokens"] == 7_000
    assert result.rows[0]["share_pct"] == 100.0


def test_timeline_rejects_unknown_period(burn_ds):
    with pytest.raises(ValueError, match="Unknown period"):
        analytics.timeline(burn_ds, "anthropic", by="quarter")


def test_timeline_no_dated_prompts():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test")
    result = analytics.timeline(empty, "anthropic")
    assert result.rows == []
    assert any("No dated prompts" in note for note in result.notes)


# ---------------------------------------------------------------------------
# 3.1 Plan break-even: API-equivalent vs subscription.
# ---------------------------------------------------------------------------


def _break_even_ds():
    """One opus prompt worth $5 of API on a single day (monthly projection $150)."""
    sessions = [{"session_id": "s1", "start_date": "2026-06-01", "project": "alpha"}]
    prompts = [
        {
            "session_id": "s1",
            "prompt_id": "p1",
            "prompt_index": 1,
            "timestamp": "2026-06-01T10:00:00Z",
            "project": "alpha",
            "model": OPUS,
        }
    ]
    tokens = [_tok("s1", "p1", OPUS, "input", 1_000_000)]  # $5.00
    return Dataset(sessions=sessions, prompts=prompts, tokens=tokens, categories={}, source="t")


def test_break_even_projects_and_ranks_plans():
    result = analytics.break_even(_break_even_ds())
    rows = _rows_by(result, "plan")
    # Bundled plans, sorted cheapest first.
    assert [r["plan"] for r in result.rows] == ["Claude Pro", "Claude Max 5x", "Claude Max 20x"]
    pro = rows["Claude Pro"]
    assert pro["monthly_price_usd"] == 20.0
    assert pro["api_equiv_month_usd"] == 150.0  # $5/day * 30
    assert pro["vs_plan"] == 7.5  # 150 / 20
    assert pro["saving_month_usd"] == 130.0
    # Best-value verdict points at the most-saving plan.
    assert any("Claude Pro" in note and "pays for itself" in note for note in result.notes)
    # Fallback note when no quota snapshots are supplied.
    assert any("No quota snapshots yet" in note for note in result.notes)


def test_break_even_no_plan_pays_off_for_tiny_usage():
    sessions = [{"session_id": "s1", "start_date": "2026-06-01", "project": "a"}]
    prompts = [
        {
            "session_id": "s1",
            "prompt_id": "p1",
            "prompt_index": 1,
            "timestamp": "2026-06-01T10:00:00Z",
            "project": "a",
            "model": OPUS,
        }
    ]
    tokens = [_tok("s1", "p1", OPUS, "input", 1_000)]  # $0.005 -> $0.15/month
    ds = Dataset(sessions=sessions, prompts=prompts, tokens=tokens, categories={}, source="t")
    result = analytics.break_even(ds)
    assert all(r["saving_month_usd"] < 0 for r in result.rows)
    assert any("no plan pays off" in note for note in result.notes)


def test_monthly_api_equivalent_projects_to_month():
    """One $5 opus prompt on a single day projects to $150/month on either grid."""
    ds = _break_even_ds()
    assert analytics.monthly_api_equivalent(ds, "anthropic") == pytest.approx(150.0)
    # Copilot prices opus identically to Anthropic, so the same $150/month.
    assert analytics.monthly_api_equivalent(ds, "copilot") == pytest.approx(150.0)


def test_copilot_channel_costs_adds_overage_over_allowance():
    """Effective cost = subscription + overage over the AI-credit allowance, cheapest first."""
    rows = analytics.copilot_channel_costs(_break_even_ds())
    by_label = {r["label"]: r for r in rows}
    # $150/month of usage; Max ($100, $200 incl) fully covers it -> flat $100.
    assert rows[0]["label"] == "Copilot Max"
    assert by_label["Copilot Max"]["overage_usd"] == pytest.approx(0.0)
    assert by_label["Copilot Max"]["total_usd"] == pytest.approx(100.0)
    # Pro+ ($39 + max(0, 150-70)=80) -> $119; Pro ($10 + 135) -> $145.
    assert by_label["Copilot Pro+"]["total_usd"] == pytest.approx(119.0)
    assert by_label["Copilot Pro"]["total_usd"] == pytest.approx(145.0)
    # Sorted cheapest first.
    assert [r["total_usd"] for r in rows] == sorted(r["total_usd"] for r in rows)


def test_break_even_enriches_with_quota_peaks():
    quota = [
        {"snapshot_at": "2026-06-01T00:00:00Z", "field": "seven_day", "utilization_pct": "40"},
        {"snapshot_at": "2026-06-02T00:00:00Z", "field": "seven_day", "utilization_pct": "75"},
        {"snapshot_at": "2026-06-02T00:00:00Z", "field": "five_hour", "utilization_pct": "12.5"},
    ]
    result = analytics.break_even(_break_even_ds(), quota_rows=quota)
    note = next(n for n in result.notes if "Quota windows" in n)
    assert "seven_day 75%" in note  # peak, not the latest 40%
    assert "five_hour 12%" in note
