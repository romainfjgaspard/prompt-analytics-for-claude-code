"""Tests for the embeddings socle (Axe B1.0).

Everything is exercised through :class:`HashingEmbedder` (the deterministic
pure-numpy double) and light fakes, so the suite never downloads the real
``model2vec`` model.
"""

from __future__ import annotations

import numpy as np
import pytest

from prompt_analytics.embeddings import (
    DEFAULT_MODEL_ID,
    Embedder,
    EmbeddingCache,
    FloatMatrix,
    HashingEmbedder,
    StaticEmbedder,
    l2_normalize,
)


def _cos(a: FloatMatrix, b: FloatMatrix) -> float:
    return float(np.dot(a, b))  # vectors are L2-normalized → dot == cosine


# ── l2_normalize ────────────────────────────────────────────────────────────


def test_l2_normalize_unit_rows_and_dtype():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_l2_normalize_zero_row_stays_zero():
    out = l2_normalize(np.array([[0.0, 0.0]]))
    np.testing.assert_array_equal(out, [[0.0, 0.0]])


def test_l2_normalize_1d_becomes_single_row():
    out = l2_normalize(np.array([3.0, 4.0]))
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)


# ── HashingEmbedder ─────────────────────────────────────────────────────────


def test_hashing_shape_and_normalized():
    emb = HashingEmbedder(dim=64)
    mat = emb.embed(["hello world", "fix the bug"])
    assert mat.shape == (2, 64)
    assert mat.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), [1.0, 1.0], atol=1e-6)


def test_hashing_is_deterministic():
    a = HashingEmbedder(dim=48).embed(["refactor this module"])
    b = HashingEmbedder(dim=48).embed(["refactor this module"])
    np.testing.assert_array_equal(a, b)


def test_hashing_empty_input():
    out = HashingEmbedder(dim=32).embed([])
    assert out.shape == (0, 32)


def test_hashing_empty_text_is_zero_vector():
    out = HashingEmbedder(dim=16).embed([""])
    np.testing.assert_array_equal(out, np.zeros((1, 16), dtype=np.float32))


def test_hashing_shared_vocabulary_is_more_similar():
    emb = HashingEmbedder(dim=512)
    near, far, query = emb.embed(
        [
            "please add unit tests for the parser",  # shares vocab with query
            "deploy the service to production",  # disjoint vocab
            "add unit tests for the parser please",  # the query (reordered)
        ]
    )
    assert _cos(query, near) > _cos(query, far)


def test_hashing_different_seeds_differ():
    a = HashingEmbedder(dim=64, seed=0).embed(["same text"])
    b = HashingEmbedder(dim=64, seed=1).embed(["same text"])
    assert not np.allclose(a, b)


def test_hashing_rejects_nonpositive_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_hashing_satisfies_embedder_protocol():
    assert isinstance(HashingEmbedder(), Embedder)


# ── EmbeddingCache ──────────────────────────────────────────────────────────


class _CountingEmbedder:
    """Wraps an embedder and counts how many texts it actually embedded."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = inner.name
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> FloatMatrix:
        self.embedded.extend(texts)
        return self.inner.embed(texts)


def test_cache_returns_correct_vectors(tmp_path):
    emb = HashingEmbedder(dim=64)
    cache = EmbeddingCache(tmp_path / "embeddings.npz", emb)
    ids = ["p1", "p2"]
    texts = ["first prompt", "second prompt"]
    got = cache.embed(ids, texts)
    np.testing.assert_array_equal(got, emb.embed(texts))


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "embeddings.npz"
    counter = _CountingEmbedder(HashingEmbedder(dim=32))
    EmbeddingCache(path, counter).embed(["p1"], ["hello"])
    assert counter.embedded == ["hello"]
    assert path.exists()

    # A fresh cache over a fresh counter must read from disk, not re-embed.
    counter2 = _CountingEmbedder(HashingEmbedder(dim=32))
    out = EmbeddingCache(path, counter2).embed(["p1"], ["hello"])
    assert counter2.embedded == []
    np.testing.assert_array_equal(out, HashingEmbedder(dim=32).embed(["hello"]))


def test_cache_only_embeds_misses(tmp_path):
    counter = _CountingEmbedder(HashingEmbedder(dim=32))
    cache = EmbeddingCache(tmp_path / "e.npz", counter)
    cache.embed(["p1"], ["a"])
    cache.embed(["p1", "p2"], ["a", "b"])  # p1/"a" is a hit
    assert counter.embedded == ["a", "b"]


def test_cache_dedupes_within_batch(tmp_path):
    counter = _CountingEmbedder(HashingEmbedder(dim=32))
    cache = EmbeddingCache(tmp_path / "e.npz", counter)
    out = cache.embed(["p1", "p1"], ["same", "same"])
    assert counter.embedded == ["same"]  # embedded once
    np.testing.assert_array_equal(out[0], out[1])


def test_cache_invalidated_by_changed_text(tmp_path):
    path = tmp_path / "e.npz"
    counter = _CountingEmbedder(HashingEmbedder(dim=32))
    EmbeddingCache(path, counter).embed(["p1"], ["original"])
    # Same prompt_id, different text → cache miss (key includes the text hash).
    EmbeddingCache(path, counter).embed(["p1"], ["edited"])
    assert counter.embedded == ["original", "edited"]


def test_cache_namespaced_by_embedder(tmp_path):
    path = tmp_path / "e.npz"
    EmbeddingCache(path, HashingEmbedder(dim=32, seed=0)).embed(["p1"], ["x"])
    # A different embedder must not reuse the first one's vectors.
    counter = _CountingEmbedder(HashingEmbedder(dim=32, seed=1))
    out = EmbeddingCache(path, counter).embed(["p1"], ["x"])
    assert counter.embedded == ["x"]
    np.testing.assert_array_equal(out, HashingEmbedder(dim=32, seed=1).embed(["x"]))


def test_cache_empty_input(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.npz", HashingEmbedder(dim=32))
    assert cache.embed([], []).shape[0] == 0


def test_cache_rejects_length_mismatch(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.npz", HashingEmbedder(dim=32))
    with pytest.raises(ValueError):
        cache.embed(["p1"], ["a", "b"])


def test_cache_rebuilds_on_corrupt_file(tmp_path):
    path = tmp_path / "e.npz"
    path.write_bytes(b"not a real npz")
    counter = _CountingEmbedder(HashingEmbedder(dim=32))
    out = EmbeddingCache(path, counter).embed(["p1"], ["x"])
    assert counter.embedded == ["x"]  # corrupt cache ignored, recomputed
    assert out.shape == (1, 32)


# ── StaticEmbedder (no network: lazy + fakes) ───────────────────────────────


class _FakeModel:
    dim = 4

    def encode(self, sentences, show_progress_bar=False):  # noqa: ARG002
        # Distinct, non-normalized vectors so we can check normalization.
        return np.array([[float(len(s)), 1.0, 2.0, 3.0] for s in sentences])


def test_static_construction_is_lazy():
    emb = StaticEmbedder()
    assert emb.model_id == DEFAULT_MODEL_ID
    assert emb.name == f"static:{DEFAULT_MODEL_ID}"
    assert emb._model is None  # nothing loaded until embed()


def test_static_satisfies_embedder_protocol():
    assert isinstance(StaticEmbedder(), Embedder)


def test_static_embed_normalizes(monkeypatch):
    emb = StaticEmbedder()
    monkeypatch.setattr(emb, "_load", lambda: _FakeModel())
    out = emb.embed(["ab", "abcd"])
    assert out.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_static_embed_empty(monkeypatch):
    emb = StaticEmbedder()
    monkeypatch.setattr(emb, "_load", lambda: _FakeModel())
    assert emb.embed([]).shape == (0, 4)


def test_static_is_cached_local_path(tmp_path):
    assert StaticEmbedder._is_cached(str(tmp_path)) is True
    assert StaticEmbedder._is_cached("definitely/not/a/real/model-xyz") is False


def test_static_load_failure_is_friendly(monkeypatch):
    """A failed fetch surfaces a clear, actionable error rather than a raw stack."""
    import model2vec

    def _boom(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(model2vec.StaticModel, "from_pretrained", _boom)
    monkeypatch.setattr(StaticEmbedder, "_is_cached", staticmethod(lambda _id: True))
    emb = StaticEmbedder("some/model")
    with pytest.raises(RuntimeError, match="vendored local copy"):
        emb.embed(["x"])
