"""Semantic embeddings — the single door to the vector model (Axe B1).

The semantic layer (mono-label classifier B1, taxonomy audit B1.2, task
clustering B2) all read prompt text as **vectors in one shared space**. This
module is that space, and the *only* place the embedding model is touched.

Design, decided 2026-06-20 (see ``PLAN-categorisation-et-composition.md`` §2):

* **Static, light, multilingual, at the core — no torch.** The model is a
  ``model2vec``-class static embedder (distilled from a multilingual
  sentence-transformer): a few tens of MB, **pure numpy at inference**, FR + EN
  in one vector space, offline after a one-time fetch. This keeps the semantic
  features available *by default* rather than hidden behind a heavy extra. The
  validation ceiling on prompt intent is ~0.7 (human agreement), so the extra
  robustness of a torch model would be largely below the task's noise floor.
* The :class:`Embedder` ``Protocol`` leaves the door open for a "max quality"
  sentence-transformer backend later (behind an optional extra) — **not built
  here**.
* :class:`HashingEmbedder` is a deterministic, pure-numpy **test double**: it
  lets the whole classification / clustering logic be exercised in CI without
  ever downloading the real model. Vectors share direction when texts share
  vocabulary, so cosine similarity behaves sensibly for tests.
* :class:`EmbeddingCache` persists vectors to ``embeddings.npz`` keyed by
  ``prompt_id`` + a hash of the text, so a re-run never re-embeds unchanged
  prompts. The cache is namespaced by embedder identity, so static and hashing
  vectors can never be mixed.

Every embedder returns an **L2-normalized** matrix (one row per input text), so
a plain dot product *is* cosine similarity downstream.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

# One row per text, unit L2 norm, float32: the shared contract of every embedder.
FloatMatrix = npt.NDArray[np.float32]

__all__ = [
    "Embedder",
    "StaticEmbedder",
    "HashingEmbedder",
    "EmbeddingCache",
    "l2_normalize",
    "DEFAULT_MODEL_ID",
]

# A model2vec multilingual static model: tens of MB on disk, pure numpy at
# inference, FR + EN in one space. Overridable (HF id or a vendored local path).
DEFAULT_MODEL_ID = "minishlab/potion-multilingual-128M"

# Word-ish chunks, lowercased: deterministic, language-agnostic enough for the
# hashing double (it only needs a stable bag of tokens to seed vectors from).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def l2_normalize(matrix: npt.NDArray[Any]) -> FloatMatrix:
    """Return ``matrix`` as float32 rows scaled to unit L2 norm.

    A 1-D input is treated as a single row. Zero rows (e.g. empty text) are left
    at zero rather than dividing by zero, so a dot product with them is 0.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    result: FloatMatrix = (arr / norms).astype(np.float32, copy=False)
    return result


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to a matrix of L2-normalized row vectors (one per text).

    ``name`` identifies the backend; the cache is namespaced by it so vectors
    from different embedders are never confused.
    """

    name: str

    def embed(self, texts: list[str]) -> FloatMatrix:
        """Embed ``texts`` → shape ``(len(texts), dim)``, L2-normalized."""
        ...


def _tokenize(text: str) -> list[str]:
    """Lowercased word-ish tokens; stable across processes."""
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Deterministic, pure-numpy stand-in for the real model (tests / CI).

    Each text is the sum of one pseudo-random unit-ish vector per token, where a
    token's vector is seeded by a stable hash of the token (``hashlib``, *not*
    Python's salted ``hash``) → reproducible across processes. Texts that share
    vocabulary point in similar directions, so cosine similarity, the τ
    threshold, and clustering can all be tested without the heavy model.
    """

    def __init__(self, dim: int = 256, *, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.seed = seed
        self.name = f"hashing:dim={dim}:seed={seed}"

    def _token_vector(self, token: str) -> npt.NDArray[np.float64]:
        digest = hashlib.blake2b(
            f"{self.seed}:{token}".encode(), digest_size=8
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        return rng.standard_normal(self.dim)

    def embed(self, texts: list[str]) -> FloatMatrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for token in _tokenize(text):
                rows[i] += self._token_vector(token)
        return l2_normalize(rows)


class StaticEmbedder:
    """The real backend: a ``model2vec`` static model (no torch, offline).

    The model is loaded lazily on the first :meth:`embed` so construction stays
    cheap and a process that only ever uses the cache or the hashing double
    never imports ``model2vec``. ``model_id`` may be a Hugging Face id or a
    vendored local directory. After the one-time fetch the model is served from
    the local Hugging Face cache (set ``HF_HOME`` to relocate it).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.name = f"static:{model_id}"
        self._model: object | None = None

    @staticmethod
    def _is_cached(model_id: str) -> bool:
        """True if the model already sits in the local HF cache (or is a path)."""
        if Path(model_id).exists():
            return True
        try:
            from huggingface_hub import try_to_load_from_cache

            # tokenizer.json is always part of a model2vec model; a real path
            # back means we can load fully offline.
            return isinstance(try_to_load_from_cache(model_id, "tokenizer.json"), str)
        except Exception:
            # Missing hub, an invalid id, ... → treat as "not cached" (the load
            # path will surface a friendly error if it really can't be fetched).
            return False

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - exercised via message only
            raise RuntimeError(
                "The static embedding model needs `model2vec` (a light, "
                "torch-free core dependency). Install it with "
                "`pip install model2vec`."
            ) from exc

        if not self._is_cached(self.model_id):
            print(
                f"Fetching the static embedding model '{self.model_id}' "
                "(one time, a few tens of MB) — it is then served offline from "
                "your local cache.",
                file=sys.stderr,
            )
        try:
            # force_download defaults to True in model2vec, which would re-fetch
            # on every run; we want the cached copy.
            self._model = StaticModel.from_pretrained(self.model_id, force_download=False)
        except Exception as exc:  # network down and not cached, bad id, ...
            raise RuntimeError(
                f"Could not load the static embedding model '{self.model_id}'. "
                "It needs a one-time download; connect to the network once, or "
                "point `model_id` at a vendored local copy. "
                f"(underlying error: {exc})"
            ) from exc
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        return int(model.dim)  # type: ignore[attr-defined]

    def embed(self, texts: list[str]) -> FloatMatrix:
        model = self._load()
        if not texts:
            return np.zeros((0, int(model.dim)), dtype=np.float32)  # type: ignore[attr-defined]
        vectors = model.encode(list(texts), show_progress_bar=False)  # type: ignore[attr-defined]
        return l2_normalize(vectors)


def _text_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def _cache_key(prompt_id: str, text: str) -> str:
    # NUL can't appear in a prompt_id, so it's an unambiguous separator.
    return f"{prompt_id}\x00{_text_hash(text)}"


class EmbeddingCache:
    """Disk-backed embedding store keyed by ``prompt_id`` + text hash.

    Wraps any :class:`Embedder`: :meth:`embed` returns vectors for the requested
    prompts, computing only the misses and persisting the enlarged cache to a
    single ``.npz``. The cache records the embedder's ``name``; if a stored file
    was written by a *different* embedder it is ignored (never mixed), so
    switching backends can't silently serve stale vectors.
    """

    def __init__(self, path: str | Path, embedder: Embedder) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self._index: dict[str, int] = {}
        self._matrix: FloatMatrix | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                if str(data["embedder"]) != self.embedder.name:
                    return  # written by another embedder → start fresh
                keys = data["keys"]
                matrix = data["matrix"]
        except Exception:
            return  # corrupt / unreadable cache → rebuild silently
        self._matrix = np.asarray(matrix, dtype=np.float32)
        self._index = {str(k): i for i, k in enumerate(keys.tolist())}

    def _save(self) -> None:
        if self._matrix is None:
            return
        keys = np.array(
            sorted(self._index, key=lambda k: self._index[k]), dtype=np.str_
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(
                handle,
                matrix=self._matrix,
                keys=keys,
                embedder=np.array(self.embedder.name, dtype=np.str_),
            )
        tmp.replace(self.path)

    def embed(self, prompt_ids: Sequence[str], texts: Sequence[str]) -> FloatMatrix:
        """Return L2-normalized vectors for ``texts`` (aligned to the inputs).

        Cache hits are reused; misses are embedded once, appended, and the file
        is rewritten. ``prompt_ids`` and ``texts`` must be the same length.
        """
        if len(prompt_ids) != len(texts):
            raise ValueError("prompt_ids and texts must have the same length")
        keys = [_cache_key(pid, text) for pid, text in zip(prompt_ids, texts, strict=True)]

        # Unique cache misses, preserving first-seen order (a batch may repeat a
        # prompt/text pair; we embed it only once).
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in self._index and key not in missing:
                missing[key] = text

        if missing:
            new_vectors = self.embedder.embed(list(missing.values()))
            self._append(list(missing.keys()), new_vectors)
            self._save()

        if not keys:
            dim = 0 if self._matrix is None else self._matrix.shape[1]
            return np.zeros((0, dim), dtype=np.float32)
        assert self._matrix is not None  # misses were just embedded → populated
        stacked: FloatMatrix = np.stack([self._matrix[self._index[key]] for key in keys])
        return stacked

    def _append(self, keys: list[str], vectors: npt.NDArray[Any]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._matrix is None:
            self._matrix = vectors
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        base = len(self._index)
        for offset, key in enumerate(keys):
            self._index[key] = base + offset
