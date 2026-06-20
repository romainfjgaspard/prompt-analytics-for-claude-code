"""Tests for the analytics layer (joins, aggregations, read-time costs).

Every aggregation is verified against hand-computed values on a small
synthetic dataset. Pricing used (bundled pricing.yml, per 1M tokens):

* claude-opus-4-8 / anthropic: input 5, output 25, cache_read 0.5,
  cache_write_5m 6.25, cache_write_1h 10.
* claude-opus-4-8 / copilot: same except cache_write_1h 6.25.
* claude-haiku-4-5 / anthropic & copilot: input 1, output 5, cache_read 0.1.
"""

from __future__ import annotations

import os

import pytest

from prompt_analytics import analytics
from prompt_analytics.analytics import CostEngine, Dataset
from prompt_analytics.extract import run_extract

OPUS = "claude-opus-4-8"
HAIKU = "claude-haiku-4-5"


def _token(session_id, prompt_id, model, token_type, count, side=0):
    return {
        "session_id": session_id,
        "prompt_id": prompt_id,
        "model": model,
        "token_type": token_type,
        "is_sidechain": side,
        "token_count": count,
    }


@pytest.fixture
def ds() -> Dataset:
    """Two sessions, three prompts, one continuation overhead bucket.

    Hand-computed anthropic costs:

    * p1 (opus): 1M input ($5) + 100k output ($2.50) + 2M cache_read ($1)
      + 400k cw_5m ($2.50) + 100k cw_1h ($1) = **$12.00** (copilot: $11.625,
      the 1h write is billed 6.25 instead of 10). 500k of the cache reads
      come from a sidechain row (subagent share: $0.25, sums unchanged).
    * p2 (opus): 200k input ($1) + 40k output ($1) + 5 server_tool_use
      requests ($0.05, anthropic per_request $0.01/req -- copilot has none) =
      **$2.05** (copilot **$2.00**). Requests stay out of the token totals.
    * p3 (haiku): 500k input ($0.50) + 100k output ($0.50) = **$1.00**.
    * s2 continuation overhead (opus): 1M cache_read = **$0.50**.

    Totals: anthropic **$15.55**, copilot **$15.125**.
    """
    sessions = [
        {
            "session_id": "s1",
            "start_date": "2026-05-01",
            "project": "alpha",
            "cwd": "/home/u/alpha",
            "git_branch": "main",
        },
        {
            "session_id": "s2",
            "start_date": "2026-05-02",
            "project": "beta",
            "cwd": "/home/u/beta",
            "git_branch": "dev",
        },
    ]
    prompts = [
        {
            "session_id": "s1",
            "prompt_id": "p1",
            "prompt_index": 1,
            "timestamp": "2026-05-01T10:00:00Z",
            "project": "alpha",
            "model": OPUS,
            "char_count": 30,
            "assistant_turns": 2,
            "tool_calls": 1,
            "prompt_preview": "implement the parser",
        },
        {
            "session_id": "s1",
            "prompt_id": "p2",
            "prompt_index": 2,
            "timestamp": "2026-05-01T11:00:00Z",
            "project": "alpha",
            "model": OPUS,
            "char_count": 10,
            "assistant_turns": 1,
            "tool_calls": 0,
            "prompt_preview": "fix the tests",
        },
        {
            "session_id": "s2",
            "prompt_id": "p3",
            "prompt_index": 1,
            "timestamp": "2026-05-02T09:00:00Z",
            "project": "beta",
            "model": HAIKU,
            "char_count": 20,
            "assistant_turns": 1,
            "tool_calls": 0,
            "prompt_preview": "what does this do",
        },
    ]
    tokens = [
        _token("s1", "p1", OPUS, "input", 1_000_000),
        _token("s1", "p1", OPUS, "output", 100_000),
        _token("s1", "p1", OPUS, "cache_read", 1_500_000),
        _token("s1", "p1", OPUS, "cache_read", 500_000, side=1),
        _token("s1", "p1", OPUS, "cache_write_5m", 400_000),
        _token("s1", "p1", OPUS, "cache_write_1h", 100_000),
        _token("s1", "p2", OPUS, "input", 200_000),
        _token("s1", "p2", OPUS, "output", 40_000),
        _token("s1", "p2", OPUS, "server_tool_use", 5),
        _token("s2", "p3", HAIKU, "input", 500_000),
        _token("s2", "p3", HAIKU, "output", 100_000),
        _token("s2", "s2:_continuation", OPUS, "cache_read", 1_000_000),
    ]
    categories = {
        "p1": {"category": "implementation", "complexity": "3"},
        "p2": {"category": "implementation", "complexity": "5"},
    }
    return Dataset(
        sessions=sessions,
        prompts=prompts,
        tokens=tokens,
        categories=categories,
        source="test data",
    )


# ---------------------------------------------------------------------------
# Cost engine.
# ---------------------------------------------------------------------------


def test_cost_engine_hand_computed_rates():
    engine = CostEngine("anthropic")
    assert engine.cost(OPUS, "input", 1_000_000) == 5.0
    assert engine.cost(OPUS, "output", 100_000) == 2.5
    assert engine.cost(OPUS, "cache_read", 2_000_000) == 1.0
    assert engine.cost(OPUS, "cache_write_5m", 400_000) == 2.5
    assert engine.cost(OPUS, "cache_write_1h", 100_000) == 1.0
    assert engine.unpriced == set()


def test_cost_engine_server_tool_use_priced_per_request():
    # 3.3: server_tool_use counts requests, billed from the provider per_request
    # table ($0.01/request for anthropic), independent of the model.
    engine = CostEngine("anthropic")
    assert engine.cost(OPUS, "server_tool_use", 5) == 0.05
    assert engine.cost(HAIKU, "server_tool_use", 5) == 0.05
    # A provider without a per_request entry leaves the requests uncosted.
    assert CostEngine("copilot").cost(OPUS, "server_tool_use", 5) == 0.0


def test_cost_engine_long_context_priced_at_base_rate_loudly():
    # 3.2: a [1m] suffix is stripped for lookup (base rate), but tracked so the
    # base-rate pricing is surfaced rather than silent.
    engine = CostEngine("anthropic")
    assert engine.cost(f"{OPUS}[1m]", "input", 1_000_000) == 5.0
    note = engine.long_context_note()
    assert note is not None and "base rate" in note and f"{OPUS}[1m]" in note


def test_cost_engine_unpriced_model_is_tracked_loudly():
    engine = CostEngine("anthropic")
    assert engine.cost("mystery-9", "input", 1_000_000) == 0.0
    assert engine.unpriced == {"mystery-9"}
    note = engine.note()
    assert note is not None and "mystery-9" in note and "WARNING" in note


def test_cost_engine_synthetic_model_is_not_reported():
    engine = CostEngine("anthropic")
    assert engine.cost("<synthetic>", "input", 10) == 0.0
    assert engine.cost("", "input", 10) == 0.0
    assert engine.unpriced == set()
    assert engine.note() is None


# ---------------------------------------------------------------------------
# Aggregations (hand-computed values).
# ---------------------------------------------------------------------------


def _rows_by(result, key):
    return {row[key]: row for row in result.rows}


def test_summary_totals(ds):
    result = analytics.summary(ds, providers=["anthropic", "copilot"])
    metrics = _rows_by(result, "metric")
    assert metrics["Sessions"]["value"] == 2
    assert metrics["Prompts"]["value"] == 3
    assert metrics["Projects"]["value"] == 2
    assert metrics["Period"]["value"] == "2026-05-01 .. 2026-05-02 (2 days)"
    assert metrics["Input tokens"]["value"] == "1,700,000"
    assert metrics["Output tokens"]["value"] == "240,000"
    assert metrics["Cache read tokens"]["value"] == "3,000,000"
    assert metrics["Cache write (5m) tokens"]["value"] == "400,000"
    assert metrics["Cache write (1h) tokens"]["value"] == "100,000"
    assert metrics["Server tool use (requests)"]["value"] == "5"
    # Total excludes the 5 server_tool_use requests.
    assert metrics["Total tokens"]["value"] == "5,440,000"
    assert metrics["Cost (anthropic)"]["value"] == "$15.55"  # incl. $0.05 server tools
    assert metrics["Cost (copilot)"]["value"] == "$15.12"  # 15.125, copilot has no per_request
    # Subagents first-class (1.2): 500k sidechain cache reads = $0.25.
    assert metrics["Subagents"]["value"] == "$0.25 (1.6% of anthropic cost)"


def test_summary_without_sidechain_rows_has_no_subagent_line(ds):
    ds.tokens = [row for row in ds.tokens if not row.get("is_sidechain")]
    result = analytics.summary(ds, providers=["anthropic"])
    assert "Subagents" not in _rows_by(result, "metric")


def test_by_project_hand_computed(ds):
    result = analytics.by_project(ds, "anthropic")
    assert [row["project"] for row in result.rows] == ["alpha", "beta"]
    alpha, beta = result.rows
    assert alpha["prompts"] == 2
    assert alpha["tokens"] == 3_840_000  # p1 3.6M + p2 240k, server requests excluded
    assert alpha["cost_usd"] == 14.05  # p1 12 + p2 2.05 (incl. $0.05 server tools)
    assert alpha["share_pct"] == 90.4  # 14.05 / 15.55
    assert alpha["token_share_pct"] == 70.6  # 3.84M / 5.44M
    # beta inherits the continuation overhead via the session join.
    assert beta["prompts"] == 1
    assert beta["tokens"] == 1_600_000
    assert beta["cost_usd"] == 1.5
    assert beta["share_pct"] == 9.6
    assert beta["token_share_pct"] == 29.4  # 1.6M / 5.44M


def test_filter_dates_no_bounds_is_identity(ds):
    assert analytics.filter_dates(ds, None, None) is ds


def test_filter_dates_since_keeps_later_prompts(ds):
    # p1/p2 are dated 2026-05-01, p3 is 2026-05-02 (session s2).
    narrowed = analytics.filter_dates(ds, "2026-05-02", None)
    assert [p["prompt_id"] for p in narrowed.prompts] == ["p3"]
    assert {s["session_id"] for s in narrowed.sessions} == {"s2"}
    # p3's tokens plus s2's continuation overhead (pseudo-prompt rides its session).
    pids = {t["prompt_id"] for t in narrowed.tokens}
    assert pids == {"p3", "s2:_continuation"}


def test_filter_dates_until_drops_other_session(ds):
    narrowed = analytics.filter_dates(ds, None, "2026-05-01")
    assert {p["prompt_id"] for p in narrowed.prompts} == {"p1", "p2"}
    assert {s["session_id"] for s in narrowed.sessions} == {"s1"}
    # s2's continuation overhead is gone with its session.
    assert all(t["session_id"] == "s1" for t in narrowed.tokens)


def test_filter_dates_empty_range(ds):
    narrowed = analytics.filter_dates(ds, "2030-01-01", None)
    assert narrowed.prompts == []
    assert narrowed.tokens == []
    assert narrowed.sessions == []


def test_by_project_cumulative_is_default(ds):
    # The cumulative %% column is now always present (no --pareto flag needed).
    result = analytics.by_project(ds, "anthropic")
    assert [row["cumulative_pct"] for row in result.rows] == [90.4, 100.0]
    assert result.columns[-1].key == "cumulative_pct"


def test_by_model_hand_computed(ds):
    result = analytics.by_model(ds, "anthropic")
    models = _rows_by(result, "model")
    opus, haiku = models[OPUS], models[HAIKU]
    assert opus["cost_usd"] == 14.55  # p1 12 + p2 2.05 + overhead 0.5
    assert opus["prompts"] == 2  # the continuation pseudo-prompt is not a prompt
    assert opus["input"] == 1_200_000
    assert opus["output"] == 140_000
    assert opus["cache_read"] == 3_000_000
    # TTL split kept apart (1.3): 1h writes are billed 2x, never merged.
    assert opus["cache_write_5m"] == 400_000
    assert opus["cache_write_1h"] == 100_000
    # Subagent split (1.2): p1's 500k sidechain cache reads = $0.25.
    assert opus["subagent_cost_usd"] == 0.25
    assert haiku["cost_usd"] == 1.0
    assert haiku["prompts"] == 1
    assert haiku["subagent_cost_usd"] == 0.0
    assert opus["share_pct"] == 93.6  # 14.55 / 15.55


def test_by_token_type_hand_computed(ds):
    """Anthropic costs per type: input $6.50 (p1 $5 + p2 $1 + p3 $0.50),
    output $4.00 ($2.50 + $1.00 + $0.50), cache_read $1.50 (3M x 0.5),
    cache_write_5m $2.50, cache_write_1h $1.00, server tools $0.05 (5 x $0.01)
    -- total $15.55."""
    result = analytics.by_token_type(ds, "anthropic")
    types = _rows_by(result, "token_type")

    assert types["Input"]["tokens"] == 1_700_000
    assert types["Input"]["cost_usd"] == 6.5
    assert types["Input"]["cost_share_pct"] == 41.8
    assert types["Output"]["cost_usd"] == 4.0
    assert types["Cache read"]["tokens"] == 3_000_000
    assert types["Cache read"]["cost_usd"] == 1.5
    # Token share is volume, not cost: cache read is 3M / 5.44M = 55.1% of tokens.
    assert types["Cache read"]["token_share_pct"] == 55.1
    assert types["Cache write (5m)"]["cost_usd"] == 2.5
    assert types["Cache write (1h)"]["cost_usd"] == 1.0

    # Rows sorted by cost, server_tool_use shown and now priced per request (3.3).
    assert [row["token_type"] for row in result.rows][:2] == ["Input", "Output"]
    server = types["Server tool use (requests, billed separately)"]
    assert server["tokens"] == 5 and server["cost_usd"] == 0.05
    # server_tool_use is counted in requests, not tokens: no token share.
    assert server["token_share_pct"] is None
    total = result.rows[-1]
    assert total["token_type"] == "TOTAL"
    assert total["tokens"] == 5_440_000  # server requests excluded from the token total
    assert total["cost_usd"] == 15.55
    assert total["token_share_pct"] == 100.0

    # The headline: context rent = (1.5 + 2.5 + 1.0) / 15.55 = 32.2%.
    assert any("Context rent" in note and "32.2%" in note for note in result.notes)


def test_by_token_type_empty_dataset_has_no_crash():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test")
    result = analytics.by_token_type(empty, "anthropic")
    assert result.rows[-1]["token_type"] == "TOTAL"
    assert result.rows[-1]["cost_usd"] == 0.0


def test_by_category_hand_computed(ds):
    result = analytics.by_category(ds, "anthropic")
    categories = _rows_by(result, "category")
    implementation = categories["implementation"]
    assert implementation["prompts"] == 2
    assert implementation["cost_usd"] == 14.05  # p1 12 + p2 2.05 (incl. server tools)
    assert implementation["avg_complexity"] == 4.0  # (3 + 5) / 2
    assert implementation["share_pct"] == 66.7
    assert implementation["cost_share_pct"] == 93.4  # 14.05 / 15.05
    # Median cost per prompt exposes intrinsically expensive categories: p1 12 + p2 2.05.
    assert implementation["med_cost_per_prompt_usd"] == 7.025
    uncategorized = categories["(uncategorized)"]
    assert uncategorized["prompts"] == 1
    assert uncategorized["cost_usd"] == 1.0
    assert uncategorized["cost_share_pct"] == 6.6  # 1.0 / 15.05
    assert uncategorized["med_cost_per_prompt_usd"] == 1.0
    # The $0.50 overhead is excluded but surfaced in the notes.
    assert any("0.50" in note for note in result.notes)


def test_by_category_without_categorization_suggests_categorize(ds):
    ds.categories = {}
    result = analytics.by_category(ds, "anthropic")
    assert [row["category"] for row in result.rows] == ["(uncategorized)"]
    assert any("categorize" in note for note in result.notes)


def test_top_prompts_hand_computed(ds):
    result = analytics.top_prompts(ds, "anthropic", top=2)
    assert [row["cost_usd"] for row in result.rows] == [12.0, 2.05]
    first = result.rows[0]
    assert first["preview"] == "implement the parser"
    assert first["project"] == "alpha"
    assert first["model"] == OPUS
    assert first["tokens"] == 3_600_000
    assert first["date"] == "2026-05-01"


def test_sessions_table_hand_computed(ds):
    result = analytics.sessions_table(ds, "anthropic", top=0)
    assert [row["session_id"] for row in result.rows] == ["s1", "s2"]
    s1, s2 = result.rows
    assert s1["cost_usd"] == 14.05  # p1 12 + p2 2.05 (incl. server tools)
    assert s1["prompts"] == 2
    assert s1["tokens"] == 3_840_000
    assert s2["cost_usd"] == 1.5  # p3 + continuation overhead
    assert s2["project"] == "beta"

    top1 = analytics.sessions_table(ds, "anthropic", top=1)
    assert len(top1.rows) == 1


def test_session_depth_hand_computed(ds):
    result = analytics.session_depth(ds, "anthropic")
    by_depth = _rows_by(result, "depth")
    # Depth 1 = p1 ($12) + p3 ($1) -> avg $6.50.
    d1 = by_depth["1"]
    assert d1["prompts"] == 2
    assert d1["avg_cost_usd"] == 6.5
    assert d1["vs_depth_1"] == 1.0
    # Input side at depth 1: 1.5M input + 2M cache_read + 500k cache_write = 4M.
    assert d1["cache_read_pct"] == 50.0
    # TTL split in the mix too (1.3): 400k 5m + 100k 1h over 4M input-side.
    assert d1["cache_write_5m_pct"] == 10.0
    assert d1["cache_write_1h_pct"] == 2.5
    assert d1["fresh_input_pct"] == 37.5
    # Per-turn normalization: depth 1 = $13 / 3 turns (p1: 2, p3: 1), and
    # 2M cache reads / 3 turns (the approximate carried context).
    assert d1["cost_per_turn_usd"] == round(13.0 / 3, 4)
    assert d1["cache_read_per_turn"] == 666_666
    # Depth 2 = p2 ($2.05, incl. $0.05 server tools) -> x0.32 vs depth 1.
    d2 = by_depth["2"]
    assert d2["prompts"] == 1
    assert d2["avg_cost_usd"] == 2.05
    assert d2["vs_depth_1"] == 0.32
    assert d2["cost_per_turn_usd"] == 2.05  # 1 turn
    assert d2["cache_read_per_turn"] == 0
    assert any("$/turn" in note for note in result.notes)
    # Continuation overhead has no prompt_index: never enters the analysis.
    assert sum(row["prompts"] for row in result.rows) == 3


def test_compare_providers_hand_computed(ds):
    result = analytics.compare_providers(ds, ["anthropic", "copilot"])
    rows = _rows_by(result, "model")
    total = rows["TOTAL"]
    assert total["cost_anthropic_usd"] == 15.55  # incl. $0.05 server tools (anthropic only)
    assert total["cost_copilot_usd"] == 15.125
    assert total["tokens"] == 5_440_000
    assert rows[OPUS]["tokens"] == 4_840_000
    assert rows[OPUS]["cost_copilot_usd"] == 14.125
    assert any("x0.97" in note for note in result.notes)  # 15.125 / 15.55
    # These are per-token API prices: point at break-even for flat-rate plans.
    assert any("break-even" in note and "per-token API" in note for note in result.notes)


def test_flat_export_hand_computed(ds):
    columns, rows = analytics.flat_export(ds, providers=["anthropic", "copilot"])
    assert len(rows) == 4  # 3 prompts + 1 overhead row
    by_pid = {row["prompt_id"]: row for row in rows}

    p1 = by_pid["p1"]
    assert p1["input_tokens"] == 1_000_000
    assert p1["cache_write_5m_tokens"] == 400_000
    assert p1["total_tokens"] == 3_600_000
    # Sidechain usage stays inside the prompt totals AND is exposed (1.2).
    assert p1["subagent_tokens"] == 500_000
    assert p1["cost_anthropic_usd"] == 12.0
    assert p1["cost_copilot_usd"] == 11.625
    assert p1["category"] == "implementation"
    assert p1["session_start_date"] == "2026-05-01"
    assert by_pid["p2"]["subagent_tokens"] == 0

    overhead = by_pid["s2:_continuation"]
    assert overhead["prompt_index"] == ""
    assert overhead["model"] == OPUS
    assert overhead["project"] == "beta"  # duplicated from the session
    assert overhead["cost_anthropic_usd"] == 0.5

    # Totals reconcile with summary.
    assert sum(row["cost_anthropic_usd"] for row in rows) == 15.55
    assert columns[-1] == "prompt_preview"
    assert set(columns) >= {"input_tokens", "output_tokens", "cost_anthropic_usd"}


def test_mini_summary_lines(ds):
    lines = analytics.mini_summary(ds, providers=["anthropic", "copilot"])
    assert "anthropic $15.55" in lines[0]
    assert "copilot $15.12" in lines[0]
    assert "alpha $14.05" in lines[1]


# ---------------------------------------------------------------------------
# Dataset loading: on-the-fly mode with output/ as a fresh cache (7.2).
# ---------------------------------------------------------------------------


def test_load_dataset_live_without_extract(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    loaded = analytics.load_dataset(fake_claude.out)
    assert "live parse" in loaded.source
    assert len(loaded.prompts) == 2
    assert all(isinstance(row["token_count"], int) for row in loaded.tokens)


def test_load_dataset_live_keeps_previews(fake_claude):
    """A live parse writes nothing to disk, so it can keep the previews:
    `prompts --top` must stay actionable without a fresh extract (D1)."""
    fake_claude.add("session_alpha.jsonl", project="alpha")
    loaded = analytics.load_dataset(fake_claude.out)
    assert "live parse" in loaded.source
    assert all(row.get("prompt_preview") for row in loaded.prompts)


def test_load_dataset_uses_fresh_csvs(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out)
    loaded = analytics.load_dataset(fake_claude.out)
    assert "(fresh)" in loaded.source
    assert len(loaded.prompts) == 2
    assert all(isinstance(row["token_count"], int) for row in loaded.tokens)


def test_load_dataset_windowed_csvs_say_so(fake_claude):
    """A --since extract is a PARTIAL cache: the Source line must say it (1.5)."""
    fake_claude.add("session_alpha.jsonl", project="alpha")
    fake_claude.add("session_beta.jsonl", project="beta")
    run_extract(fake_claude.out, since="2026-05-10", timezone_name="UTC")
    loaded = analytics.load_dataset(fake_claude.out)
    assert "(fresh, window: since 2026-05-10)" in loaded.source

    # A full extract clears the marker.
    run_extract(fake_claude.out)
    loaded = analytics.load_dataset(fake_claude.out)
    assert "(fresh)" in loaded.source
    assert "window:" not in loaded.source


def test_dataset_from_csvs_reports_window(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out, since="2026-05-01", until="2026-05-02", timezone_name="UTC")
    ds = analytics.dataset_from_csvs(fake_claude.out)
    assert "window: since 2026-05-01, until 2026-05-02" in ds.source


def test_load_dataset_stale_csvs_reparse(fake_claude):
    path = fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out)
    os.utime(path, (path.stat().st_atime + 60, path.stat().st_mtime + 60))
    loaded = analytics.load_dataset(fake_claude.out)
    assert "live parse" in loaded.source
    # The user pointed at existing CSVs that were NOT used: say so loudly.
    assert "IGNORED" in loaded.source


def test_load_dataset_old_schema_csvs_reparse(fake_claude):
    """A pre-phase-7 tokens.csv (no model column) is never trusted."""
    fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out)
    tokens_path = fake_claude.out / "tokens.csv"
    lines = tokens_path.read_text(encoding="utf-8").splitlines()
    legacy = ["session_id,prompt_id,token_type,token_count"]
    legacy += [",".join(line.split(",")[:2] + line.split(",")[3:]) for line in lines[1:]]
    tokens_path.write_text("\n".join(legacy) + "\n", encoding="utf-8")
    # Make the CSVs newer than the JSONL so only the schema check can reject.
    loaded = analytics.load_dataset(fake_claude.out)
    assert "live parse" in loaded.source


def test_load_dataset_no_cache_forces_reparse(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    run_extract(fake_claude.out)
    loaded = analytics.load_dataset(fake_claude.out, use_cache=False)
    assert "live parse" in loaded.source
    # Deliberate bypass (--no-cache), not a staleness surprise: no warning.
    assert "IGNORED" not in loaded.source


def test_load_dataset_reads_categories_in_live_mode(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    fake_claude.out.mkdir(parents=True, exist_ok=True)
    (fake_claude.out / "categories.csv").write_text(
        "prompt_id,category,complexity,classifier_model,classified_at\n"
        "pA1,debug,2,test-model,2026-06-01T00:00:00Z\n",
        encoding="utf-8",
    )
    loaded = analytics.load_dataset(fake_claude.out)
    assert loaded.categories == {"pA1": {"category": "debug", "complexity": "2"}}


def test_live_dataset_costs_match_csv_dataset(fake_claude):
    """The two loading paths must produce identical aggregations."""
    fake_claude.add("session_alpha.jsonl", project="alpha")
    fake_claude.add("session_beta.jsonl", project="beta")
    live = analytics.load_dataset(fake_claude.out)
    run_extract(fake_claude.out)
    cached = analytics.load_dataset(fake_claude.out)
    assert "live parse" in live.source and "(fresh)" in cached.source

    live_rows = analytics.by_project(live, "anthropic").rows
    cached_rows = analytics.by_project(cached, "anthropic").rows
    assert live_rows == cached_rows


# ---------------------------------------------------------------------------
# by_output: output composition (Axe C, C2).
# ---------------------------------------------------------------------------


def _output_ds() -> Dataset:
    """Two prompts with file edits + a prose/code output split.

    Hand-computed (anthropic): opus output $25/1M, haiku output $5/1M.

    * p1 (opus): 100k output ($2.50), split prose 60k / code 40k
      -> prose $1.50, code $1.00. Files: Python code (+100/-10, 2 files),
      Python test (+50, 1 file).
    * p3 (haiku): 100k output ($0.50), split prose 20k / code 80k
      -> prose $0.10, code $0.40. Files: SQL code (+20, 1 file).

    Lines added: Python 150 (test 50), SQL 20; total 170 (test 50 = 29.4%).
    """
    sessions = [{"session_id": "s1", "project": "alpha"}]
    prompts = [
        {"session_id": "s1", "prompt_id": "p1", "project": "alpha", "model": OPUS},
        {"session_id": "s1", "prompt_id": "p3", "project": "alpha", "model": HAIKU},
    ]
    tokens = [
        _token("s1", "p1", OPUS, "output", 100_000),
        _token("s1", "p3", HAIKU, "output", 100_000),
    ]
    output_files = [
        # p1 edits two Python code files (100/+10 total) and one Python test file.
        {
            "prompt_id": "p1",
            "path": "src/a.py",
            "language": "Python",
            "kind": "code",
            "edits": 1,
            "lines_added": 50,
            "lines_deleted": 5,
        },
        {
            "prompt_id": "p1",
            "path": "src/b.py",
            "language": "Python",
            "kind": "code",
            "edits": 1,
            "lines_added": 50,
            "lines_deleted": 5,
        },
        {
            "prompt_id": "p1",
            "path": "tests/t.py",
            "language": "Python",
            "kind": "test",
            "edits": 1,
            "lines_added": 50,
            "lines_deleted": 0,
        },
        {
            "prompt_id": "p3",
            "path": "q.sql",
            "language": "SQL",
            "kind": "code",
            "edits": 1,
            "lines_added": 20,
            "lines_deleted": 0,
        },
    ]
    output_tokens = [
        {"prompt_id": "p1", "output_prose_tokens": 60_000, "output_code_tokens": 40_000},
        {"prompt_id": "p3", "output_prose_tokens": 20_000, "output_code_tokens": 80_000},
    ]
    return Dataset(
        sessions=sessions,
        prompts=prompts,
        tokens=tokens,
        categories={},
        source="test data",
        output_files=output_files,
        output_tokens=output_tokens,
    )


def test_by_output_language_mix_and_kind_split():
    result = analytics.by_output(_output_ds(), "anthropic")
    by_lang = {r["language"]: r for r in result.rows}

    python = by_lang["Python"]
    assert python["files"] == 3
    assert python["lines_added"] == 150
    assert python["lines_deleted"] == 10
    assert python["test_pct"] == round(100 * 50 / 150, 1)  # 33.3
    assert python["share_pct"] == round(100 * 150 / 170, 1)  # 88.2

    sql = by_lang["SQL"]
    assert sql["lines_added"] == 20 and sql["test_pct"] == 0.0

    total = by_lang["TOTAL"]
    assert total["files"] == 4
    assert total["lines_added"] == 170
    assert total["lines_deleted"] == 10
    assert total["test_pct"] == round(100 * 50 / 170, 1)  # 29.4
    assert total["share_pct"] == 100.0


def test_by_output_sorted_by_lines_with_total_last():
    rows = analytics.by_output(_output_ds(), "anthropic").rows
    assert [r["language"] for r in rows] == ["Python", "SQL", "TOTAL"]


def test_by_output_prose_vs_code_cost_note():
    notes = analytics.by_output(_output_ds(), "anthropic").notes
    # prose $1.50+$0.10=$1.60, code $1.00+$0.40=$1.40 -> code = 46.7% of $3.00.
    gen_note = next(n for n in notes if n.startswith("Generated output"))
    assert "$1.60" in gen_note and "$1.40" in gen_note
    assert "46.7% of generation cost is code" in gen_note
    assert "80,000 prose" in gen_note and "120,000 code/tool" in gen_note
    tests_note = next(n for n in notes if n.startswith("Code vs tests"))
    assert "29.4%" in tests_note and "120 code, 50 test" in tests_note


def test_by_output_empty_dataset_hints_to_extract():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test data")
    result = analytics.by_output(empty, "anthropic")
    assert result.rows == []
    assert any("No output-composition data" in n for n in result.notes)


def test_output_composition_language_costs_reconcile_to_code_cost():
    comp = analytics.output_composition(_output_ds(), "anthropic")
    # p1: $1.00 code cost, all Python (churn 110+50=160 all Python) -> Python $1.00.
    # p3: $0.40 code cost, all SQL -> SQL $0.40. No tooling (both edited files).
    by_lang = {lng.language: lng for lng in comp.languages}
    assert by_lang["Python"].code_cost == pytest.approx(1.00, abs=1e-6)
    assert by_lang["SQL"].code_cost == pytest.approx(0.40, abs=1e-6)
    assert comp.tooling_cost == pytest.approx(0.0, abs=1e-9)
    # Per-language code costs + tooling reconcile to the total code cost.
    summed = sum(lng.code_cost for lng in comp.languages) + comp.tooling_cost
    assert summed == pytest.approx(comp.code_cost, abs=1e-6)
    assert comp.code_cost == pytest.approx(1.40, abs=1e-6)
    assert comp.prose_cost == pytest.approx(1.60, abs=1e-6)


def test_output_composition_code_tokens_without_file_edit_go_to_tooling():
    # A prompt that spent output tokens on tooling (Bash/Read) but edited no file:
    # no output_files row, so its code cost lands in the tooling bucket.
    ds = Dataset(
        sessions=[{"session_id": "s1"}],
        prompts=[{"session_id": "s1", "prompt_id": "p1", "model": OPUS}],
        tokens=[_token("s1", "p1", OPUS, "output", 100_000)],
        categories={},
        source="test data",
        output_files=[],
        output_tokens=[
            {"prompt_id": "p1", "output_prose_tokens": 50_000, "output_code_tokens": 50_000}
        ],
    )
    comp = analytics.output_composition(ds, "anthropic")
    assert comp.languages == []
    assert comp.tooling_cost == pytest.approx(comp.code_cost, abs=1e-9)
    assert comp.code_cost > 0


# ---------------------------------------------------------------------------
# file_footprint: the unified per-file view (DASH4), Axe C joined to Axe D.
# ---------------------------------------------------------------------------


def _footprint_ds() -> Dataset:
    """A file edited *and* read (app.py) plus a read-only file (config.json)."""
    return Dataset(
        sessions=[{"session_id": "s1"}],
        prompts=[{"session_id": "s1", "prompt_id": "p1", "project": "a", "model": OPUS}],
        tokens=[],
        categories={},
        source="test data",
        output_files=[
            {
                "prompt_id": "p1",
                "path": "src/app.py",
                "language": "Python",
                "kind": "code",
                "edits": 3,
                "lines_added": 120,
                "lines_deleted": 40,
            }
        ],
        output_tokens=[],
        context_sources=[
            {
                "session_id": "s1",
                "source": "file_read",
                "language": "Python",
                "path": "src/app.py",
                "tokens": 5000,
                "items": 4,
            },
            {
                "session_id": "s1",
                "source": "file_read",
                "language": "JSON",
                "path": "config.json",
                "tokens": 2000,
                "items": 2,
            },
            {
                "session_id": "s1",
                "source": "conversation",
                "language": "-",
                "path": "-",
                "tokens": 9000,
                "items": 10,
            },
        ],
        context_cost=[
            {
                "session_id": "s1",
                "source": "file_read",
                "language": "Python",
                "path": "src/app.py",
                "model": OPUS,
                "rent_read_tokens": 800_000,
                "load_write_5m_tokens": 50_000,
                "load_write_1h_tokens": 0,
            },
            {
                "session_id": "s1",
                "source": "file_read",
                "language": "JSON",
                "path": "config.json",
                "model": OPUS,
                "rent_read_tokens": 400_000,
                "load_write_5m_tokens": 0,
                "load_write_1h_tokens": 0,
            },
            {
                "session_id": "s1",
                "source": "conversation",
                "language": "-",
                "path": "-",
                "model": OPUS,
                "rent_read_tokens": 300_000,
                "load_write_5m_tokens": 0,
                "load_write_1h_tokens": 0,
            },
        ],
    )


def test_file_footprint_crosses_edits_and_context_cost():
    rows = analytics.file_footprint(_footprint_ds(), "anthropic").rows
    by_path = {r["path"]: r for r in rows}
    # app.py shows BOTH halves: edits + line diff (C) and reads + context cost (D).
    app = by_path["src/app.py"]
    assert app["language"] == "Python" and app["kind"] == "code"
    assert (app["edits"], app["lines_added"], app["lines_deleted"]) == (3, 120, 40)
    assert app["reads"] == 4
    assert app["load_usd"] > 0 and app["rent_usd"] > 0
    assert app["context_usd"] == pytest.approx(app["load_usd"] + app["rent_usd"], abs=1e-9)
    # The read-only manifest has a pure D footprint (no edits) -- the cut candidate.
    cfg = by_path["config.json"]
    assert cfg["edits"] == 0 and cfg["reads"] == 2 and cfg["rent_usd"] > 0
    # conversation is not a file -> no per-file row.
    assert "-" not in by_path


def test_file_footprint_sorted_by_context_cost_with_note():
    result = analytics.file_footprint(_footprint_ds(), "anthropic")
    # app.py (more rent) sorts before config.json.
    assert [r["path"] for r in result.rows] == ["src/app.py", "config.json"]
    note = next(n for n in result.notes if "files:" in n)
    assert "1 edited" in note and "1 read but never edited" in note


def test_file_footprint_empty_dataset_hints_to_extract():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test data")
    result = analytics.file_footprint(empty, "anthropic")
    assert result.rows == []
    assert any("No per-file data" in n for n in result.notes)


# ---------------------------------------------------------------------------
# by_context: context cost over time (Axe D, D2).
# ---------------------------------------------------------------------------


def _context_cost_ds() -> Dataset:
    """Context cost rows + the matching main-chain requests (opus rates).

    opus: cache_read $0.5/1M, cache_write_5m $6.25/1M, cache_write_1h $10/1M.

    * file_read Python: rent 2M ($1.00), load_5m 1M ($6.25) -> total $7.25
    * conversation:     rent 1M ($0.50), load_1h 0.1M ($1.00) -> total $1.50
    * (unattributed):   rent 0.2M ($0.10)

    Bill: cache_read 3.2M ($1.60) + write_5m 1M ($6.25) + write_1h 0.1M ($1.00)
    = $8.85, which the attributed $8.75 + $0.10 unattributed reconciles to.
    """
    context_cost = [
        {
            "session_id": "s1",
            "source": "file_read",
            "language": "Python",
            "model": OPUS,
            "rent_read_tokens": 2_000_000,
            "load_write_5m_tokens": 1_000_000,
            "load_write_1h_tokens": 0,
        },
        {
            "session_id": "s1",
            "source": "conversation",
            "language": "-",
            "model": OPUS,
            "rent_read_tokens": 1_000_000,
            "load_write_5m_tokens": 0,
            "load_write_1h_tokens": 100_000,
        },
        {
            "session_id": "s1",
            "source": "(unattributed)",
            "language": "-",
            "model": OPUS,
            "rent_read_tokens": 200_000,
            "load_write_5m_tokens": 0,
            "load_write_1h_tokens": 0,
        },
    ]
    requests = [
        {
            "session_id": "s1",
            "prompt_id": "p1",
            "model": OPUS,
            "is_sidechain": 0,
            "cache_read_tokens": 3_200_000,
            "cache_write_5m_tokens": 1_000_000,
            "cache_write_1h_tokens": 100_000,
        },
        # A subagent request carries its own context: excluded from the bill.
        {
            "session_id": "s1",
            "prompt_id": "p1",
            "model": OPUS,
            "is_sidechain": 1,
            "cache_read_tokens": 9_000_000,
            "cache_write_5m_tokens": 0,
            "cache_write_1h_tokens": 0,
        },
    ]
    return Dataset(
        sessions=[{"session_id": "s1", "project": "alpha"}],
        prompts=[{"session_id": "s1", "prompt_id": "p1", "project": "alpha", "model": OPUS}],
        tokens=[],
        categories={},
        source="test data",
        requests=requests,
        context_cost=context_cost,
    )


def test_context_cost_splits_load_and_rent_per_element():
    comp = analytics.context_cost(_context_cost_ds(), "anthropic")
    by_src = {(e.source, e.language): e for e in comp.elements}
    python = by_src[("file_read", "Python")]
    assert python.load_cost == pytest.approx(6.25, abs=1e-6)
    assert python.rent_cost == pytest.approx(1.00, abs=1e-6)
    conv = by_src[("conversation", "-")]
    assert conv.load_cost == pytest.approx(1.00, abs=1e-6)
    assert conv.rent_cost == pytest.approx(0.50, abs=1e-6)
    # (unattributed) is kept out of the elements but folded into the residual.
    assert ("(unattributed)", "-") not in by_src
    assert comp.unattributed_cost == pytest.approx(0.10, abs=1e-6)


def test_context_cost_reconciles_to_the_billed_main_chain():
    ds = _context_cost_ds()
    comp = analytics.context_cost(ds, "anthropic")
    engine = analytics.CostEngine("anthropic")
    bill = analytics._main_chain_cache_cost(ds, engine)
    assert comp.total_cost == pytest.approx(8.85, abs=1e-6)
    assert comp.total_cost == pytest.approx(bill, abs=1e-6)  # subagent read excluded
    assert comp.attributed_cost == pytest.approx(8.75, abs=1e-6)


def test_by_context_table_sorted_with_total_and_unattributed():
    result = analytics.by_context(_context_cost_ds(), "anthropic")
    # Sorted by total cost: file_read Python ($7.25) before conversation ($1.50),
    # then the (unattributed) line, then TOTAL last.
    labels = [(r["source"], r["language"]) for r in result.rows]
    assert labels[0] == ("Files read", "Python")
    assert labels[1] == ("Conversation", "-")
    assert ("(unattributed)", "-") in labels
    assert labels[-1] == ("TOTAL", "")
    total = result.rows[-1]
    assert total["total_usd"] == pytest.approx(8.85, abs=1e-4)
    recon = next(n for n in result.notes if n.startswith("Reconciliation"))
    assert "$8.75" in recon and "$8.85" in recon


def test_by_context_empty_dataset_hints_to_extract():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test data")
    result = analytics.by_context(empty, "anthropic")
    assert result.rows == []
    assert any("No context-cost data" in n for n in result.notes)


def test_filter_prompt_ids_narrows_output_rows():
    ds = _output_ds()
    kept = analytics.filter_prompt_ids(ds, {"p1"})
    assert {r["prompt_id"] for r in kept.prompts} == {"p1"}
    assert {r["prompt_id"] for r in kept.output_files} == {"p1"}
    assert {r["prompt_id"] for r in kept.output_tokens} == {"p1"}
    assert {r["prompt_id"] for r in kept.tokens} == {"p1"}
    # The narrowed dataset feeds the composition view: only Python survives.
    comp = analytics.output_composition(kept, "anthropic")
    assert [lng.language for lng in comp.languages] == ["Python"]


# ---------------------------------------------------------------------------
# by_task: cost by task (Axe B2), the task as the unit of work.
# ---------------------------------------------------------------------------


def _task_ds() -> Dataset:
    """One session, two tasks (a todo spine + an inferred one) and a continuation.

    OPUS rates (per 1M): input 5, output 25, cache_read 0.5, cache_write_5m 6.25.
    t1 = p1 + p2: input $5 + output $5 + cache_read ($1 + $0.5) = $11.50 ($1.50 ctx).
    t2 = p3: input $2 + cache_write_5m $6.25 = $8.25 ($6.25 ctx). A continuation
    token row (no task) is the $0.50 overhead, excluded from every task.
    """
    tokens = [
        _token("s1", "p1", OPUS, "input", 1_000_000),  # $5.00
        _token("s1", "p1", OPUS, "output", 200_000),  # $5.00
        _token("s1", "p1", OPUS, "cache_read", 2_000_000),  # $1.00 (context)
        _token("s1", "p2", OPUS, "cache_read", 1_000_000),  # $0.50 (context)
        _token("s1", "p3", OPUS, "input", 400_000),  # $2.00
        _token("s1", "p3", OPUS, "cache_write_5m", 1_000_000),  # $6.25 (context)
        _token("s1", "s1:cont", OPUS, "cache_read", 1_000_000),  # $0.50 overhead
    ]
    tasks = [
        {
            "task_id": "s1:t01",
            "session_id": "s1",
            "name": "Add the export pipeline",
            "origin": "todo",
            "prompts": 2,
            "first_timestamp": "2026-06-01T08:00:00.000Z",
            "last_timestamp": "2026-06-01T10:30:00.000Z",
        },
        {
            "task_id": "s1:i01",
            "session_id": "s1",
            "name": "fix the failing test",
            "origin": "inferred",
            "prompts": 1,
            "first_timestamp": "2026-06-01T11:00:00.000Z",
            "last_timestamp": "2026-06-01T11:00:00.000Z",
        },
    ]
    task_prompts = [
        {"task_id": "s1:t01", "prompt_id": "p1"},
        {"task_id": "s1:t01", "prompt_id": "p2"},
        {"task_id": "s1:i01", "prompt_id": "p3"},
    ]
    categories = {
        "p1": {"category": "implementation", "complexity": "3"},
        "p2": {"category": "implementation", "complexity": "2"},
        "p3": {"category": "debug", "complexity": "2"},
    }
    return Dataset(
        sessions=[{"session_id": "s1", "project": "alpha"}],
        prompts=[{"session_id": "s1", "prompt_id": pid} for pid in ("p1", "p2", "p3")],
        tokens=tokens,
        categories=categories,
        source="test data",
        tasks=tasks,
        task_prompts=task_prompts,
    )


def test_by_task_cost_split_context_and_dominant_category():
    rows = {r["task"]: r for r in analytics.by_task(_task_ds(), "anthropic").rows}

    t1 = rows["Add the export pipeline"]
    assert t1["origin"] == "todo"
    assert t1["prompts"] == 2
    assert t1["duration"] == "2h 30m"
    assert t1["category"] == "implementation"
    assert t1["cost_usd"] == pytest.approx(11.50, abs=1e-6)
    assert t1["context_pct"] == round(100 * 1.50 / 11.50, 1)  # 13.0
    assert t1["cost_share_pct"] == round(100 * 11.50 / 19.75, 1)  # 58.2

    t2 = rows["fix the failing test"]
    assert t2["origin"] == "inferred"
    assert t2["duration"] == "<1m"  # single instant
    assert t2["category"] == "debug"
    assert t2["cost_usd"] == pytest.approx(8.25, abs=1e-6)
    assert t2["context_pct"] == round(100 * 6.25 / 8.25, 1)  # 75.8


def test_by_task_sorted_by_cost_and_top_truncates():
    rows = analytics.by_task(_task_ds(), "anthropic").rows
    assert [r["task"] for r in rows] == ["Add the export pipeline", "fix the failing test"]
    top1 = analytics.by_task(_task_ds(), "anthropic", top=1).rows
    assert [r["task"] for r in top1] == ["Add the export pipeline"]


def test_by_task_notes_origin_split_context_and_overhead():
    notes = analytics.by_task(_task_ds(), "anthropic").notes
    spine = next(n for n in notes if "TodoWrite spine" in n)
    assert "2 tasks across 1 sessions" in spine
    assert "1 from the TodoWrite spine, 1 inferred" in spine
    ctx = next(n for n in notes if n.startswith("Context is"))
    assert "$7.75 of $19.75" in ctx  # 1.50 + 6.25 of 11.50 + 8.25
    assert "39.2%" in ctx
    overhead = next(n for n in notes if n.startswith("Session overhead"))
    assert "$0.50" in overhead


def test_by_task_costs_reconcile_to_the_bill():
    ds = _task_ds()
    engine = CostEngine("anthropic")
    task_cost = sum(r["cost_usd"] for r in analytics.by_task(ds, "anthropic").rows)
    overhead = engine.cost(OPUS, "cache_read", 1_000_000)  # the continuation row
    total_bill = sum(engine.cost(t["model"], t["token_type"], t["token_count"]) for t in ds.tokens)
    assert task_cost + overhead == pytest.approx(total_bill, abs=1e-6)


def test_by_task_uncategorized_when_no_categories():
    ds = _task_ds()
    ds.categories.clear()
    result = analytics.by_task(ds, "anthropic")
    assert all(r["category"] == "(uncategorized)" for r in result.rows)
    assert any("No categorization found" in n for n in result.notes)


def test_by_task_empty_dataset_hints_to_extract():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test data")
    result = analytics.by_task(empty, "anthropic")
    assert result.rows == []
    assert any("No task data" in n for n in result.notes)


# ---------------------------------------------------------------------------
# task_graph: the Axe-B2 force-graph data (task centres + prompt satellites).
# ---------------------------------------------------------------------------


def test_task_graph_centres_satellites_and_totals():
    graph = analytics.task_graph(_task_ds(), "anthropic")

    # Two task centres, sorted by cost (the $11.50 todo task first).
    assert [t.name for t in graph.tasks] == ["Add the export pipeline", "fix the failing test"]
    centre = graph.tasks[0]
    assert centre.category == "implementation"
    assert centre.origin == "todo"
    assert centre.prompts == 2
    assert centre.cost == pytest.approx(11.50, abs=1e-6)
    assert centre.context_pct == round(100 * 1.50 / 11.50, 1)

    # Satellites = the prompts of the shown tasks (3 across both), each linked.
    assert {s.prompt_id for s in graph.satellites} == {"p1", "p2", "p3"}
    assert {s.task_id for s in graph.satellites} == {"s1:t01", "s1:i01"}
    p1 = next(s for s in graph.satellites if s.prompt_id == "p1")
    assert p1.category == "implementation"
    assert p1.cost == pytest.approx(11.00, abs=1e-6)  # input $5 + output $5 + cache_read $1

    # Population + reconciled totals (overhead excluded, like by_task).
    assert graph.total_tasks == 2
    assert graph.todo_tasks == 1
    assert graph.grand_total == pytest.approx(19.75, abs=1e-6)
    assert graph.context_total == pytest.approx(7.75, abs=1e-6)
    assert graph.has_data


def test_task_graph_top_limits_centres_and_their_satellites():
    graph = analytics.task_graph(_task_ds(), "anthropic", top=1)
    assert [t.task_id for t in graph.tasks] == ["s1:t01"]
    # Only the kept task's prompts orbit; the dropped task's prompt is gone.
    assert {s.prompt_id for s in graph.satellites} == {"p1", "p2"}
    # The population count still reflects every task, not just the shown slice.
    assert graph.total_tasks == 2


def test_task_graph_uncategorized_without_categories():
    ds = _task_ds()
    ds.categories.clear()
    graph = analytics.task_graph(ds, "anthropic")
    assert all(t.category == "(uncategorized)" for t in graph.tasks)
    assert all(s.category == "(uncategorized)" for s in graph.satellites)


def test_task_graph_empty_dataset_has_no_data():
    empty = Dataset(sessions=[], prompts=[], tokens=[], categories={}, source="test data")
    graph = analytics.task_graph(empty, "anthropic")
    assert graph.tasks == []
    assert graph.satellites == []
    assert not graph.has_data


# ---------------------------------------------------------------------------
# impact: before/after a date pivot (Axe E, the capstone transverse view).
# ---------------------------------------------------------------------------


def _impact_ds() -> Dataset:
    """Two prompts before a pivot, two after, with clean OPUS-priced costs.

    OPUS rates (per 1M): input 5, output 25, cache_read 0.5.
    BEFORE (s1, 2026-06-01): p1 = p2 = input 1M ($5) + output 100k ($2.50) +
    cache_read 1M ($0.50) = $8.00 each -> $16.00, output $5, context $1.
    AFTER (s2, 2026-06-10): p3 = p4 = input 2M ($10) + output 400k ($10) +
    cache_read 4M ($2) = $22.00 each -> $44.00, output $20, context $4.
    One request per prompt (the 'turns'). Pivot 2026-06-05 splits 2 / 2.
    """

    def _p(session, pid, day):
        return {"session_id": session, "prompt_id": pid, "timestamp": f"{day}T10:00:00Z"}

    def _req(session, pid):
        return {"session_id": session, "prompt_id": pid, "is_sidechain": 0}

    tokens = []
    for pid in ("p1", "p2"):
        tokens += [
            _token("s1", pid, OPUS, "input", 1_000_000),
            _token("s1", pid, OPUS, "output", 100_000),
            _token("s1", pid, OPUS, "cache_read", 1_000_000),
        ]
    for pid in ("p3", "p4"):
        tokens += [
            _token("s2", pid, OPUS, "input", 2_000_000),
            _token("s2", pid, OPUS, "output", 400_000),
            _token("s2", pid, OPUS, "cache_read", 4_000_000),
        ]
    return Dataset(
        sessions=[{"session_id": "s1"}, {"session_id": "s2"}],
        prompts=[
            _p("s1", "p1", "2026-06-01"),
            _p("s1", "p2", "2026-06-01"),
            _p("s2", "p3", "2026-06-10"),
            _p("s2", "p4", "2026-06-10"),
        ],
        tokens=tokens,
        categories={
            "p1": {"category": "implementation"},
            "p2": {"category": "implementation"},
            "p3": {"category": "debug"},
            "p4": {"category": "implementation"},
        },
        source="test data",
        requests=[_req("s1", "p1"), _req("s1", "p2"), _req("s2", "p3"), _req("s2", "p4")],
        tasks=[{"task_id": "s1:t1", "session_id": "s1"}],
        task_prompts=[{"task_id": "s1:t1", "prompt_id": "p1"}],
    )


def _metric(report, label):
    return next(m for m in report.metrics if m.label == label)


def test_impact_report_normalized_ratios_split_on_pivot():
    report = analytics.impact_report(_impact_ds(), provider="anthropic", pivot="2026-06-05")

    assert report.before_prompts == 2
    assert report.after_prompts == 2
    assert report.has_both_sides

    cpp = _metric(report, "Cost per prompt")
    assert cpp.before == pytest.approx(8.0)  # $16.00 / 2
    assert cpp.after == pytest.approx(22.0)  # $44.00 / 2

    out_share = _metric(report, "Output cost share")
    assert out_share.before == pytest.approx(100 * 5 / 16)  # 31.25
    assert out_share.after == pytest.approx(100 * 20 / 44)  # 45.45

    ctx_share = _metric(report, "Context rent share")
    assert ctx_share.before == pytest.approx(100 * 1 / 16)  # 6.25
    assert ctx_share.after == pytest.approx(100 * 4 / 44)  # 9.09

    crpt = _metric(report, "Cache read / turn")
    assert crpt.before == pytest.approx(2_000_000 / 2)  # two requests before
    assert crpt.after == pytest.approx(8_000_000 / 2)

    oppt = _metric(report, "Output tokens / prompt")
    assert oppt.before == pytest.approx(200_000 / 2)
    assert oppt.after == pytest.approx(800_000 / 2)


def test_impact_report_confounders_and_total_reconciles():
    report = analytics.impact_report(_impact_ds(), provider="anthropic", pivot="2026-06-05")

    before_total = _metric(report, "Total cost").before
    after_total = _metric(report, "Total cost").after
    assert before_total == pytest.approx(16.0)
    assert after_total == pytest.approx(44.0)
    # The two sides reconcile to the whole bill (every token is on exactly one side).
    engine = CostEngine("anthropic")
    bill = sum(
        engine.cost(t["model"], t["token_type"], t["token_count"]) for t in _impact_ds().tokens
    )
    assert before_total + after_total == pytest.approx(bill)

    prompts = _metric(report, "Prompts")
    assert prompts.confounder and prompts.before == 2 and prompts.after == 2
    pps = _metric(report, "Prompts / session")
    assert pps.before == pytest.approx(2.0) and pps.after == pytest.approx(2.0)
    cats = _metric(report, "Top category")
    assert cats.before == "implementation"  # 2 implementation
    assert cats.after == "debug"  # debug wins the 1-1 tie by Counter order (first seen)


def test_impact_table_has_confounder_divider_and_honesty_note():
    result = analytics.impact(_impact_ds(), "anthropic", pivot="2026-06-05")
    metrics = [r["metric"] for r in result.rows]
    # Ratios come first, then the divider, then the confounders.
    assert "Cost per prompt" in metrics
    divider = next(i for i, m in enumerate(metrics) if "confounders" in m)
    assert metrics.index("Cost per prompt") < divider < metrics.index("Prompts")
    # The honesty caveat (observational, not causal) is always present.
    assert any("observational split" in n for n in result.notes)
    assert any("Pivot 2026-06-05" in n for n in result.notes)
    # Percentage-point delta for a share row; signed money delta for a cost row.
    ctx_row = next(r for r in result.rows if r["metric"] == "Context rent share")
    assert ctx_row["change"].endswith("pp")
    cpp_row = next(r for r in result.rows if r["metric"] == "Cost per prompt")
    assert cpp_row["change"].startswith("+$")


def test_impact_empty_side_warns_and_blanks_cells():
    # A pivot before all the data: the BEFORE side is empty.
    result = analytics.impact(_impact_ds(), "anthropic", pivot="2026-06-01")
    report = analytics.impact_report(_impact_ds(), provider="anthropic", pivot="2026-06-01")
    assert not report.has_both_sides
    assert any(n.startswith("WARNING: no prompts before") for n in result.notes)
    cpp_row = next(r for r in result.rows if r["metric"] == "Cost per prompt")
    assert cpp_row["before"] == "-"  # nothing to normalize on the empty side
    assert cpp_row["change"] == "-"


def test_impact_drops_optional_rows_without_tasks_or_categories():
    ds = _impact_ds()
    ds.tasks.clear()
    ds.categories.clear()
    report = analytics.impact_report(ds, provider="anthropic", pivot="2026-06-05")
    labels = {m.label for m in report.metrics}
    assert "Tasks" not in labels
    assert "Top category" not in labels
    assert "Cost per prompt" in labels  # the ratios always stay


def test_split_on_pivot_partitions_the_whole():
    # The dashboard's date-pivot mode (DASH2) splits a dataset into before/after;
    # the pivot day lands in AFTER, the prior day in BEFORE, and every prompt is on
    # exactly one side.
    before, after = analytics.split_on_pivot(_impact_ds(), "2026-06-05")
    before_ids = {p["prompt_id"] for p in before.prompts}
    after_ids = {p["prompt_id"] for p in after.prompts}
    assert before_ids == {"p1", "p2"}  # dated 2026-06-01
    assert after_ids == {"p3", "p4"}  # dated 2026-06-10 (pivot day onward)
    assert before_ids.isdisjoint(after_ids)
    assert before_ids | after_ids == {"p1", "p2", "p3", "p4"}


def test_split_on_pivot_includes_pivot_day_in_after():
    # A pivot exactly on a prompt's day keeps that prompt on the AFTER side.
    before, after = analytics.split_on_pivot(_impact_ds(), "2026-06-10")
    assert {p["prompt_id"] for p in before.prompts} == {"p1", "p2"}
    assert {p["prompt_id"] for p in after.prompts} == {"p3", "p4"}


def test_impact_suggestions_folded_into_notes():
    result = analytics.impact(
        _impact_ds(),
        "anthropic",
        pivot="2026-06-05",
        suggestions=[("2026-06-03", "webapp/CLAUDE.md")],
    )
    assert any("2026-06-03 (webapp/CLAUDE.md)" in n for n in result.notes)


def test_suggest_pivots_from_config_mtime_inside_range(tmp_path):
    import os
    from datetime import datetime

    project = tmp_path / "webapp"
    project.mkdir()
    claude_md = project / "CLAUDE.md"
    claude_md.write_text("# project memory\n", encoding="utf-8")
    # Set the mtime to 2026-06-05 12:00 local, strictly inside [06-01, 06-10].
    pivot_ts = datetime(2026, 6, 5, 12, 0, 0).timestamp()
    os.utime(claude_md, (pivot_ts, pivot_ts))

    ds = _impact_ds()
    ds.sessions[0]["cwd"] = str(project)
    suggestions = analytics.suggest_pivots(ds)
    assert ("2026-06-05", "webapp/CLAUDE.md") in suggestions


def test_suggest_pivots_filters_out_of_range_dates(tmp_path):
    import os
    from datetime import datetime

    project = tmp_path / "webapp"
    project.mkdir()
    claude_md = project / "CLAUDE.md"
    claude_md.write_text("x", encoding="utf-8")
    # Mtime way after the data window -> filtered out (degenerate split).
    far = datetime(2027, 1, 1, 12, 0, 0).timestamp()
    os.utime(claude_md, (far, far))

    ds = _impact_ds()
    ds.sessions[0]["cwd"] = str(project)
    assert analytics.suggest_pivots(ds) == []
