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


def test_by_project_pareto_cumulative(ds):
    result = analytics.by_project(ds, "anthropic", pareto=True)
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
