"""Local token counting (``tiktoken`` at the core, with an offline fallback).

The prose/code output split (Axe C) prorates a message's *real* output tokens
by the token weight of its ``text`` blocks vs its ``tool_use`` blocks. We count
those weights with a **local** tokenizer so nothing ever leaves the machine.

``tiktoken`` is a light core dependency; if it is missing, or cannot load its
vocabulary (e.g. a fully offline first run with no cached encoding), a
deterministic word/punctuation heuristic takes over. The split degrades
gracefully because it only ever consumes a *ratio* of two counts -- a constant
bias in the counter cancels out -- and the result stays reproducible.

The tokenizer is deliberately not Anthropic's: the absolute counts are an
estimate (the API reports only a per-message total), but the ratio is stable.
Language detection and line diffs (the other Axe C metrics) never touch this
module -- they are exact, derived straight from the tool inputs.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

__all__ = ["count_tokens"]

# cl100k_base ships with tiktoken and loads offline once cached; it is a close
# enough stand-in for a generic BPE vocabulary (we only need a ratio).
_ENCODING = "cl100k_base"

# Word chunks + standalone punctuation: deterministic and roughly token-shaped.
_APPROX_RE = re.compile(r"\w+|[^\w\s]")


@lru_cache(maxsize=1)
def _encoder() -> Any | None:
    """Return a cached tiktoken encoder, or ``None`` if unavailable.

    Any failure (missing package, no network for the vocabulary, ...) collapses
    to ``None`` so the caller falls back to the offline heuristic.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding(_ENCODING)
    except Exception:
        return None


def _approx(text: str) -> int:
    """Offline fallback: count word-ish chunks and standalone punctuation."""
    return len(_APPROX_RE.findall(text))


def count_tokens(text: str) -> int:
    """Count the tokens of ``text`` with the local tokenizer (0 for empty).

    Uses ``tiktoken`` when available; otherwise a deterministic heuristic. Never
    raises and never sends anything anywhere.
    """
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return _approx(text)
