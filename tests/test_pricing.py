"""Tests for prompt_analytics.pricing — 11.2.

Covers: invalid YAML, unknown provider, prefix fallback, [1m] suffix,
cache invalidation between tests, and drift check against the embedded grid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_analytics.pricing import (
    PricingError,
    clear_cache,
    get_model_pricing,
    load_pricing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_valid_yml(tmp_path: Path, extra_provider: str = "") -> Path:
    extra = (
        f"""
  {extra_provider}:
    models:
      test-model:
        input: 1.0
        output: 2.0
        cache_read: 0.1
        cache_write_5m: 1.25
        cache_write_1h: 2.0
    fallbacks: {{}}
"""
        if extra_provider
        else ""
    )
    content = f"""
updated_at: "2026-01-01"
providers:
  anthropic:
    models:
      claude-test-1:
        input: 5.0
        output: 25.0
        cache_read: 0.5
        cache_write_5m: 6.25
        cache_write_1h: 10.0
    fallbacks:
      claude-test:
        input: 5.0
        output: 25.0
        cache_read: 0.5
        cache_write_5m: 6.25
        cache_write_1h: 10.0
{extra}
"""
    p = tmp_path / "pricing.yml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_pricing — validation
# ---------------------------------------------------------------------------


def test_load_pricing_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yml"
    p.write_text("{ unclosed: [", encoding="utf-8")
    with pytest.raises(PricingError, match="Invalid YAML"):
        load_pricing(p)


def test_load_pricing_missing_updated_at(tmp_path: Path) -> None:
    p = tmp_path / "bad.yml"
    p.write_text("providers: {}\n", encoding="utf-8")
    with pytest.raises(PricingError, match="updated_at"):
        load_pricing(p)


def test_load_pricing_missing_providers(tmp_path: Path) -> None:
    p = tmp_path / "bad.yml"
    p.write_text("updated_at: '2026-01-01'\n", encoding="utf-8")
    with pytest.raises(PricingError, match="providers"):
        load_pricing(p)


def test_load_pricing_providers_not_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yml"
    p.write_text("updated_at: '2026-01-01'\nproviders: not-a-mapping\n", encoding="utf-8")
    with pytest.raises(PricingError, match="providers.*mapping"):
        load_pricing(p)


def test_load_pricing_model_missing_required_key(tmp_path: Path) -> None:
    p = tmp_path / "bad.yml"
    p.write_text(
        """
updated_at: "2026-01-01"
providers:
  anthropic:
    models:
      claude-bad:
        input: 5.0
        output: 25.0
        cache_read: 0.5
        cache_write_5m: 6.25
        # missing cache_write_1h
    fallbacks: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(PricingError, match="cache_write_1h"):
        load_pricing(p)


def test_load_pricing_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match="Cannot read"):
        load_pricing(tmp_path / "nonexistent.yml")


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


def test_clear_cache_invalidates_between_loads(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path)
    load_pricing(p)
    # Mutate file on disk — without clear_cache the old object would be returned.
    new_content = p.read_text(encoding="utf-8").replace("2026-01-01", "2027-12-31")
    p.write_text(new_content, encoding="utf-8")

    cached = load_pricing(p)
    assert cached["updated_at"] == "2026-01-01"  # still cached

    clear_cache()
    fresh = load_pricing(p)
    assert fresh["updated_at"] == "2027-12-31"


# ---------------------------------------------------------------------------
# get_model_pricing — provider / model resolution
# ---------------------------------------------------------------------------


def test_unknown_provider_returns_none(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path)
    clear_cache()
    assert get_model_pricing("claude-test-1", "bedrock", pricing_path=p) is None


def test_exact_match(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path)
    clear_cache()
    entry = get_model_pricing("claude-test-1", "anthropic", pricing_path=p)
    assert entry is not None
    assert entry["input"] == 5.0
    assert entry["output"] == 25.0
    assert entry["cache_write_5m"] == 6.25
    assert entry["cache_write_1h"] == 10.0


def test_prefix_fallback(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path)
    clear_cache()
    # "claude-test-9-9" has no exact match but starts with "claude-test"
    entry = get_model_pricing("claude-test-9-9", "anthropic", pricing_path=p)
    assert entry is not None
    assert entry["input"] == 5.0


def test_prefix_fallback_longest_wins(tmp_path: Path) -> None:
    """When two fallback prefixes match, the longer one takes precedence."""
    p = tmp_path / "pricing.yml"
    p.write_text(
        """
updated_at: "2026-01-01"
providers:
  anthropic:
    models: {}
    fallbacks:
      claude:
        input: 1.0
        output: 2.0
        cache_read: 0.1
        cache_write_5m: 1.25
        cache_write_1h: 2.0
      claude-opus:
        input: 5.0
        output: 25.0
        cache_read: 0.5
        cache_write_5m: 6.25
        cache_write_1h: 10.0
""",
        encoding="utf-8",
    )
    clear_cache()
    entry = get_model_pricing("claude-opus-99", "anthropic", pricing_path=p)
    assert entry is not None
    assert entry["input"] == 5.0  # longer prefix "claude-opus" wins over "claude"


def test_unknown_model_returns_none(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path)
    clear_cache()
    assert get_model_pricing("gpt-totally-unknown", "anthropic", pricing_path=p) is None


# ---------------------------------------------------------------------------
# [1m] / long-context suffix stripping
# ---------------------------------------------------------------------------


def test_1m_suffix_stripped(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path)
    clear_cache()
    entry = get_model_pricing("claude-test-1[1m]", "anthropic", pricing_path=p)
    assert entry is not None
    assert entry["input"] == 5.0


def test_long_context_suffix_stripped(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path)
    clear_cache()
    entry = get_model_pricing("claude-test[long-context]", "anthropic", pricing_path=p)
    assert entry is not None
    assert entry["input"] == 5.0


def test_arbitrary_bracket_suffix_stripped(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path)
    clear_cache()
    entry = get_model_pricing("claude-test-1[any-suffix]", "anthropic", pricing_path=p)
    assert entry is not None


# ---------------------------------------------------------------------------
# user-defined provider is extensible
# ---------------------------------------------------------------------------


def test_custom_provider_works(tmp_path: Path) -> None:
    p = _write_valid_yml(tmp_path, extra_provider="mycompany")
    clear_cache()
    entry = get_model_pricing("test-model", "mycompany", pricing_path=p)
    assert entry is not None
    assert entry["input"] == 1.0


# ---------------------------------------------------------------------------
# Drift check — embedded grid vs. known-good reference (11.2)
#
# Values verified 2026-06-11 against
# https://docs.anthropic.com/en/docs/about-claude/pricing.
# A test failure here means pricing.yml has drifted from the published rates.
# ---------------------------------------------------------------------------

# fmt: off
REFERENCE_ANTHROPIC = {
    "claude-fable-5":            {"input": 10.00, "output": 50.00, "cache_read": 1.00,  "cache_write_5m": 12.50, "cache_write_1h": 20.00},  # noqa: E241
    "claude-opus-4-8":           {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write_5m": 6.25,  "cache_write_1h": 10.00},  # noqa: E241
    "claude-opus-4-7":           {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write_5m": 6.25,  "cache_write_1h": 10.00},  # noqa: E241
    "claude-opus-4-6":           {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write_5m": 6.25,  "cache_write_1h": 10.00},  # noqa: E241
    "claude-opus-4-5":           {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write_5m": 6.25,  "cache_write_1h": 10.00},  # noqa: E241
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00, "cache_read": 0.30,  "cache_write_5m": 3.75,  "cache_write_1h": 6.00},   # noqa: E241
    "claude-sonnet-4-5":         {"input": 3.00,  "output": 15.00, "cache_read": 0.30,  "cache_write_5m": 3.75,  "cache_write_1h": 6.00},   # noqa: E241
    "claude-sonnet-4":           {"input": 3.00,  "output": 15.00, "cache_read": 0.30,  "cache_write_5m": 3.75,  "cache_write_1h": 6.00},   # noqa: E241
    "claude-haiku-4-5":          {"input": 1.00,  "output": 5.00,  "cache_read": 0.10,  "cache_write_5m": 1.25,  "cache_write_1h": 2.00},   # noqa: E241
    "claude-haiku-4-5-20251001": {"input": 1.00,  "output": 5.00,  "cache_read": 0.10,  "cache_write_5m": 1.25,  "cache_write_1h": 2.00},   # noqa: E241
}
# fmt: on


@pytest.mark.parametrize("model,expected", REFERENCE_ANTHROPIC.items())
def test_embedded_anthropic_values(model: str, expected: dict) -> None:  # type: ignore[type-arg]
    """Embedded pricing.yml matches the verified reference grid."""
    clear_cache()
    entry = get_model_pricing(model, "anthropic")
    assert entry is not None, f"No pricing found for {model!r} / anthropic"
    for key, value in expected.items():
        assert entry[key] == pytest.approx(value), (
            f"{model} / anthropic / {key}: got {entry[key]}, expected {value}"
        )


def test_embedded_copilot_single_cache_write_mapping() -> None:
    """Copilot cache_write_5m == cache_write_1h (single-tier mapping §4.2)."""
    clear_cache()
    entry = get_model_pricing("claude-opus-4-8", "copilot")
    assert entry is not None
    assert entry["cache_write_5m"] == entry["cache_write_1h"]


def test_embedded_copilot_fable5_present() -> None:
    clear_cache()
    entry = get_model_pricing("claude-fable-5", "copilot")
    assert entry is not None
    assert entry["input"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Plans (3.1), per-request server tools (3.3), long-context flag (3.2).
# ---------------------------------------------------------------------------


def test_embedded_plans_present() -> None:
    """The bundled grid ships flat-rate subscription plans (3.1)."""
    from prompt_analytics.pricing import load_plans

    clear_cache()
    plans = load_plans()
    assert {"claude_pro", "claude_max_5x", "claude_max_20x"} <= set(plans)
    assert plans["claude_max_20x"]["monthly_usd"] == pytest.approx(200.0)
    assert plans["claude_max_5x"]["label"] == "Claude Max 5x"


def test_embedded_copilot_plans_present() -> None:
    """The bundled grid ships GitHub Copilot subscription tiers with credit allowances."""
    from prompt_analytics.pricing import load_copilot_plans

    clear_cache()
    tiers = load_copilot_plans()
    assert {
        "copilot_pro",
        "copilot_pro_plus",
        "copilot_max",
        "copilot_business",
        "copilot_enterprise",
    } <= set(tiers)
    assert tiers["copilot_pro"]["monthly_usd"] == pytest.approx(10.0)
    assert tiers["copilot_pro"]["included_usd"] == pytest.approx(15.0)
    assert tiers["copilot_max"]["included_usd"] == pytest.approx(200.0)
    assert tiers["copilot_pro_plus"]["label"] == "Copilot Pro+"


def test_embedded_per_request_web_search() -> None:
    """Anthropic per_request prices server_tool_use at $0.01/request (3.3)."""
    from prompt_analytics.pricing import get_per_request

    clear_cache()
    assert get_per_request("server_tool_use", "anthropic") == pytest.approx(0.01)
    # Copilot has no per_request table -> None (uncosted).
    assert get_per_request("server_tool_use", "copilot") is None
    assert get_per_request("unknown_tool", "anthropic") is None


def test_is_long_context_flags_known_suffixes() -> None:
    from prompt_analytics.pricing import is_long_context

    assert is_long_context("claude-opus-4-8[1m]")
    assert is_long_context("claude-opus-4-8[long-context]")
    assert is_long_context("claude-sonnet-4-6[200k+]")
    assert not is_long_context("claude-opus-4-8")
    assert not is_long_context("claude-opus-4-8[thinking]")


def test_long_context_suffix_still_priced_at_base_rate() -> None:
    """A [1m] model strips to base for lookup (3.2)."""
    clear_cache()
    entry = get_model_pricing("claude-opus-4-8[1m]", "anthropic")
    assert entry is not None and entry["input"] == pytest.approx(5.0)


def test_invalid_plan_missing_monthly_usd_raises(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yml"
    path.write_text(
        'updated_at: "2026-06-12"\n'
        "plans:\n"
        "  bad_plan:\n"
        '    label: "No price"\n'
        "providers:\n"
        "  anthropic:\n"
        "    models:\n"
        "      m:\n"
        "        input: 1\n        output: 1\n        cache_read: 1\n"
        "        cache_write_5m: 1\n        cache_write_1h: 1\n",
        encoding="utf-8",
    )
    clear_cache()
    with pytest.raises(PricingError, match="monthly_usd"):
        load_pricing(path)


def test_invalid_per_request_non_numeric_raises(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yml"
    path.write_text(
        'updated_at: "2026-06-12"\n'
        "providers:\n"
        "  anthropic:\n"
        "    per_request:\n"
        '      server_tool_use: "free"\n'
        "    models:\n"
        "      m:\n"
        "        input: 1\n        output: 1\n        cache_read: 1\n"
        "        cache_write_5m: 1\n        cache_write_1h: 1\n",
        encoding="utf-8",
    )
    clear_cache()
    with pytest.raises(PricingError, match="per_request"):
        load_pricing(path)
