"""Guards on the committed demo dataset and its generator (plan v2, 1.8).

The demo powers the public dashboard and the README screenshots; the audit 07
measured drifting totals between two loads, so determinism and internal
consistency are pinned here:

* two ``generate()`` calls produce identical rows (seeded, no hidden state);
* the committed CSVs are in sync with the generator (no stale regeneration);
* the committed data satisfies the schema-v2 invariants the dashboard relies
  on: per-prompt request sums equal tokens.csv (V7/V10), sidechain and
  post-compaction dimensions are represented, pseudo-prompt rows exist (N1).
"""

from __future__ import annotations

import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo_data"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_demo_data", REPO_ROOT / "scripts" / "generate_demo_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv(name: str) -> list[dict[str, str]]:
    with (DEMO_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def generated():
    return _load_generator().generate()


def test_generator_is_deterministic(generated):
    again = _load_generator().generate()
    assert generated == again


def test_committed_demo_matches_the_generator(generated):
    """A generator change must come with a regenerated demo_data/."""
    tokens = _read_csv("tokens.csv")
    requests = _read_csv("requests.csv")
    assert len(tokens) == len(generated["tokens"])
    assert len(requests) == len(generated["requests"])
    assert sum(int(r["token_count"]) for r in tokens) == sum(
        int(str(r["token_count"])) for r in generated["tokens"]
    )


def test_demo_requests_sums_match_tokens_per_prompt():
    """V7/V10 on the committed data: the request grain never drifts."""
    col_of = {
        "input": "input_tokens",
        "output": "output_tokens",
        "cache_read": "cache_read_tokens",
        "cache_write_5m": "cache_write_5m_tokens",
        "cache_write_1h": "cache_write_1h_tokens",
        "server_tool_use": "server_tool_use_requests",
    }
    request_sums: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in _read_csv("requests.csv"):
        for token_type, col in col_of.items():
            request_sums[row["prompt_id"]][token_type] += int(row[col])

    token_sums: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in _read_csv("tokens.csv"):
        token_sums[row["prompt_id"]][row["token_type"]] += int(row["token_count"])

    assert set(request_sums) == set(token_sums)
    for pid, counts in token_sums.items():
        for token_type, count in counts.items():
            assert request_sums[pid][token_type] == count, (pid, token_type)


def test_committed_output_composition_matches_generator(generated):
    """output_files.csv / output_tokens.csv stay in sync with the generator."""
    assert len(_read_csv("output_files.csv")) == len(generated["output_files"])
    assert len(_read_csv("output_tokens.csv")) == len(generated["output_tokens"])


def test_demo_output_split_reconciles_with_output_tokens():
    """Axe C invariant on the committed demo: prose + code == prompt output."""
    output_by_prompt: dict[str, int] = defaultdict(int)
    for row in _read_csv("tokens.csv"):
        if row["token_type"] == "output" and row["is_sidechain"] == "0":
            output_by_prompt[row["prompt_id"]] += int(row["token_count"])

    split = _read_csv("output_tokens.csv")
    assert split, "the demo must ship an output_tokens.csv"
    for row in split:
        prose = int(row["output_prose_tokens"])
        code = int(row["output_code_tokens"])
        assert prose + code == output_by_prompt[row["prompt_id"]], row["prompt_id"]


def test_demo_output_files_are_metrics_only():
    """output_files.csv carries relative paths + metrics only (no source code)."""
    prompt_ids = {r["prompt_id"] for r in _read_csv("prompts.csv")}
    rows = _read_csv("output_files.csv")
    assert rows
    for row in rows:
        # File rows exist for real prompts only, and carry positive metrics.
        assert row["prompt_id"] in prompt_ids
        assert row["kind"] in ("code", "test")
        assert int(row["lines_added"]) >= 0 and int(row["edits"]) >= 1
        # The path is a project-relative identity (never absolute).
        assert row["path"] and not row["path"].startswith("/")


def test_committed_context_composition_matches_generator(generated):
    """context_sources.csv / context_cost.csv stay in sync with the generator."""
    assert len(_read_csv("context_sources.csv")) == len(generated["context_sources"])
    assert len(_read_csv("context_cost.csv")) == len(generated["context_cost"])


def test_demo_context_cost_reconciles_with_billed_cache():
    """Axe D invariant: attributed rent/load == billed main-chain cache, to the token."""
    main = [r for r in _read_csv("requests.csv") if r["is_sidechain"] == "0"]
    cost = _read_csv("context_cost.csv")
    assert cost, "the demo must ship a context_cost.csv"
    assert sum(int(r["rent_read_tokens"]) for r in cost) == sum(
        int(r["cache_read_tokens"]) for r in main
    )
    assert sum(int(r["load_write_5m_tokens"]) for r in cost) == sum(
        int(r["cache_write_5m_tokens"]) for r in main
    )
    assert sum(int(r["load_write_1h_tokens"]) for r in cost) == sum(
        int(r["cache_write_1h_tokens"]) for r in main
    )


def test_demo_context_is_metrics_only():
    """No content / file paths leak into the Axe D CSVs (privacy guard)."""
    for name in ("context_sources.csv", "context_cost.csv"):
        rows = _read_csv(name)
        assert rows
        for row in rows:
            assert "/" not in row.get("session_id", "")
    # File languages are labels, never paths or content.
    langs = {r["language"] for r in _read_csv("context_sources.csv")}
    assert "Python" in langs or "Markdown" in langs
    assert not any("." in lang and "/" in lang for lang in langs)


def test_demo_represents_the_new_dimensions():
    tokens = _read_csv("tokens.csv")
    requests = _read_csv("requests.csv")
    prompt_ids = {r["prompt_id"] for r in _read_csv("prompts.csv")}

    # Subagents (1.2), TTL split (1.3), compaction (1.4).
    assert any(r["is_sidechain"] == "1" for r in tokens)
    assert any(r["token_type"] == "cache_write_1h" for r in tokens)
    assert any(r["post_compact"] == "1" for r in requests)
    # Pseudo-prompt rows (N1): in tokens.csv, absent from prompts.csv.
    pseudo = {r["prompt_id"] for r in tokens} - prompt_ids
    assert pseudo and all(":_continuation" in pid for pid in pseudo)
