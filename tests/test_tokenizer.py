"""Tests for the local token counter (tiktoken at the core, offline fallback)."""

from __future__ import annotations

from prompt_analytics import tokenizer


def test_count_tokens_empty_is_zero():
    assert tokenizer.count_tokens("") == 0


def test_count_tokens_is_positive_and_monotonic():
    short = tokenizer.count_tokens("hello world")
    longer = tokenizer.count_tokens("hello world, this is a longer sentence with code()")
    assert short > 0
    assert longer > short


def test_fallback_used_when_encoder_unavailable(monkeypatch):
    """With no tiktoken encoder, the deterministic heuristic still counts."""
    monkeypatch.setattr(tokenizer, "_encoder", lambda: None)
    # Word chunks + standalone punctuation: "a", "b", "(", ")" -> 4.
    assert tokenizer.count_tokens("a b()") == 4
    assert tokenizer.count_tokens("") == 0


def test_encoder_failure_degrades_gracefully(monkeypatch):
    class _Boom:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("vocab unreachable")

    monkeypatch.setattr(tokenizer, "_encoder", lambda: _Boom())
    # Falls back to the heuristic instead of raising.
    assert tokenizer.count_tokens("a b") == 2
