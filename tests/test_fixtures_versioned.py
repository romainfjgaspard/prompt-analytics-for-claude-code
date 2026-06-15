"""Pin parsing against versioned, anonymized real-format fixtures (11.8).

Each ``tests/fixtures/claude-code-<version>/`` directory is an anonymized copy
of a real Claude Code session (captured via ``scripts/capture_fixture.py``).
Walking them here is the canary the ccusage maintainers recommend: if an
upstream format change breaks parsing, one of these fails instead of silently
under-counting real usage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_analytics import extract

FIXTURES = Path(__file__).parent / "fixtures"
VERSION_DIRS = sorted(
    p for p in FIXTURES.glob("claude-code-*") if p.is_dir() and any(p.rglob("*.jsonl"))
)


def test_at_least_one_versioned_fixture_exists() -> None:
    assert VERSION_DIRS, "expected at least one claude-code-<version>/ fixture"


@pytest.mark.parametrize("version_dir", VERSION_DIRS, ids=lambda p: p.name)
def test_versioned_fixture_parses_cleanly(version_dir: Path) -> None:
    result = extract.collect(use_cache=False, claude_dir=version_dir)
    report = result.report

    # The format still parses: real prompts and token usage come out.
    assert report.prompts >= 1
    assert result.tokens, "expected at least one token row"
    assert sum(t["token_count"] for t in result.tokens) > 0

    # No corruption and no unrecognized event types (the format-drift canary).
    assert report.lines_invalid == 0
    assert report.unknown_event_types == {}

    # The directory name advertises the version actually present in the data.
    declared = version_dir.name.removeprefix("claude-code-")
    assert declared in report.versions


@pytest.mark.parametrize("version_dir", VERSION_DIRS, ids=lambda p: p.name)
def test_versioned_fixture_parse_is_deterministic(version_dir: Path) -> None:
    a = extract.collect(use_cache=False, claude_dir=version_dir)
    b = extract.collect(use_cache=False, claude_dir=version_dir)
    assert a.report.prompts == b.report.prompts
    assert sorted((t["token_type"], t["token_count"]) for t in a.tokens) == sorted(
        (t["token_type"], t["token_count"]) for t in b.tokens
    )
