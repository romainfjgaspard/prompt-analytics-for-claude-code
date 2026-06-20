"""Tests for the dev-time semantic eval/calibration tool (Axe B1.3).

Everything runs offline: the :class:`HashingEmbedder` stands in for the real
model and a fake client stands in for the LLM judge, so there is **no network**
in the suite. The committed litmus fixture is validated for shape, and the pure
helpers (agreement matrix, calibration grid, judge caching, report rendering)
are pinned on small constructed inputs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from prompt_analytics.categorize import CATEGORIES
from prompt_analytics.embeddings import HashingEmbedder

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_eval() -> Any:
    spec = importlib.util.spec_from_file_location(
        "eval_semantic", REPO_ROOT / "scripts" / "eval_semantic.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses (3.10) resolve field types via
    # sys.modules[cls.__module__]; an unregistered module makes that lookup None.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ev() -> Any:
    return _load_eval()


# ── committed fixture shape ─────────────────────────────────────────────────────


def test_litmus_fixture_is_valid(ev: Any) -> None:
    cases = ev.load_litmus()
    assert len(cases) >= 15  # a real litmus, not a stub
    for c in cases:
        assert c.text.strip(), "empty litmus prompt"
        assert c.gold in CATEGORIES, f"litmus gold {c.gold!r} not in taxonomy"
    # the test/implementation boundary (the archi complaint) must be represented
    golds = {c.gold for c in cases}
    assert {"test", "implementation"} <= golds


def test_load_demo_texts_skips_pseudo(ev: Any) -> None:
    texts = ev.load_demo_texts()
    assert texts, "demo corpus should not be empty"
    assert all(":_" not in pid for pid in texts), "pseudo prompts must be excluded"
    assert all(t.strip() for t in texts.values())


# ── agreement matrix ────────────────────────────────────────────────────────────


def test_agreement_matrix_counts_and_rate(ev: Any) -> None:
    a = {"p1": "debug", "p2": "test", "p3": "other", "p4": "plan"}
    b = {"p1": "debug", "p2": "plan", "p3": "implementation", "p4": "plan"}
    res = ev.agreement_matrix(a, b)
    assert res.n_total == 4
    assert res.n_agree == 2  # p1 and p4
    assert res.agreement == 0.5
    assert res.matrix[("test", "plan")] == 1
    assert res.matrix[("debug", "debug")] == 1


def test_agreement_matrix_ignores_unshared_ids(ev: Any) -> None:
    res = ev.agreement_matrix({"p1": "debug"}, {"p2": "debug"})
    assert res.n_total == 0
    assert res.agreement == 0.0


# ── divergence sampling ─────────────────────────────────────────────────────────


def test_sample_divergences_buckets_and_dedup(ev: Any) -> None:
    texts = {
        "p1": "fix the crash",
        "p2": "fix the crash",  # duplicate text → must appear once
        "p3": "design the schema",
        "p4": "agree with both",
    }
    heur = {"p1": "other", "p2": "other", "p3": "plan", "p4": "debug"}
    sem = {"p1": "debug", "p2": "debug", "p3": "refactor", "p4": "debug"}
    rescued, contradictions = ev.sample_divergences(texts, heur, sem)
    assert rescued == [("fix the crash", "debug")]  # deduped, p4 agrees so excluded
    assert contradictions == [("design the schema", "plan", "refactor")]


# ── litmus evaluation ───────────────────────────────────────────────────────────


def test_evaluate_litmus_accuracy(ev: Any) -> None:
    cases = [
        ev.LitmusCase(text="a", gold="debug"),
        ev.LitmusCase(text="b", gold="test"),
        ev.LitmusCase(text="c", gold="plan"),
    ]
    res = ev.evaluate_litmus(cases, ["debug", "review", "plan"])
    assert res.accuracy == pytest.approx(2 / 3)
    assert [ok for *_r, ok in res.rows] == [True, False, True]


# ── calibration grid ────────────────────────────────────────────────────────────


def test_calibrate_picks_grid_max_litmus(ev: Any) -> None:
    from prompt_analytics.categorize import SemanticClassifier

    clf = SemanticClassifier(HashingEmbedder(seed=1))
    cases = ev.load_litmus()
    golds = [c.gold for c in cases]
    preps, vectors = ev.prepare_and_embed(clf, [c.text for c in cases])

    cal = ev.calibrate(clf, preps, vectors, golds)
    # the chosen point is genuinely the grid-max on litmus accuracy
    assert cal.best_litmus_acc == max(lit for _t, _p, lit, _j in cal.grid)
    assert (cal.best_tau, cal.best_prime_weight) in {(t, p) for t, p, _l, _j in cal.grid}
    assert cal.grid, "grid must be non-empty"
    assert cal.used_judge is False
    # evaluating without a judge leaves the classifier's knobs restored
    assert clf.tau == ev.DEFAULT_TAU and clf.prime_weight == ev.DEFAULT_PRIME_WEIGHT


def test_calibrate_uses_judge_as_tiebreak(ev: Any) -> None:
    from prompt_analytics.categorize import SemanticClassifier

    clf = SemanticClassifier(HashingEmbedder(seed=2))
    cases = ev.load_litmus()[:6]
    golds = [c.gold for c in cases]
    preps, vectors = ev.prepare_and_embed(clf, [c.text for c in cases])
    jpreps, jvectors = ev.prepare_and_embed(clf, ["foo bar", "baz qux"])

    cal = ev.calibrate(
        clf,
        preps,
        vectors,
        golds,
        judge_preps=jpreps,
        judge_vectors=jvectors,
        judge_golds=["debug", "test"],
    )
    assert cal.used_judge is True
    assert cal.best_judge_agree is not None


# ── LLM judge (fake client, no network) ─────────────────────────────────────────


class _FakeJudge:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls = 0

    def classify(self, text: str) -> tuple[str, str]:
        self.calls += 1
        return self.mapping.get(text, "other"), "3"


def test_run_judge_caches_by_text(ev: Any, tmp_path: Path) -> None:
    client = _FakeJudge({"hello": "question", "world": "debug"})
    items = [("p1", "hello"), ("p2", "world")]
    cache = tmp_path / "judge.json"

    first = ev.run_judge(client, items, cache_path=cache)
    assert first == {"p1": "question", "p2": "debug"}
    assert client.calls == 2
    assert cache.exists()
    assert set(json.loads(cache.read_text(encoding="utf-8")).values()) == {"question", "debug"}

    # second run reuses the cache → no new client calls
    second = ev.run_judge(client, items, cache_path=cache)
    assert second == first
    assert client.calls == 2


def test_run_judge_skips_blank_replies(ev: Any) -> None:
    client = _FakeJudge({"a": ""})  # empty category → row skipped, not "other"
    out = ev.run_judge(client, [("p1", "a")], cache_path=None)
    assert out == {}


def test_judge_accuracies(ev: Any) -> None:
    judge = {"p1": "debug", "p2": "test", "p3": "plan"}
    heur = {"p1": "debug", "p2": "test", "p3": "other"}
    sem = {"p1": "debug", "p2": "refactor", "p3": "plan"}
    res = ev.judge_accuracies(judge, heur, sem)
    assert res.n == 3
    assert res.heuristic_acc == pytest.approx(2 / 3)
    assert res.semantic_acc == pytest.approx(2 / 3)


# ── end-to-end report (hashing embedder + fake judge) ───────────────────────────


def test_build_and_render_report(ev: Any) -> None:
    texts = {
        f"s:p{i}": t
        for i, t in enumerate(
            ["fix the crash", "add tests for the parser", "commit and push", "what is uv lock"]
        )
    }
    litmus = ev.load_litmus()[:8]
    judge = _FakeJudge({t: "debug" for t in texts.values()})

    report = ev.build_report(
        embedder=HashingEmbedder(seed=3),
        texts=texts,
        litmus=litmus,
        judge_client=judge,
        judge_sample=10,
        judge_cache=None,
    )
    assert report.n_prompts == 4
    assert report.judge is not None and report.judge.n == 4
    assert 0.0 <= report.agreement.agreement <= 1.0

    rendered = ev.render_report(report)
    for header in ("LITMUS SET", "HEURISTIC vs SEMANTIC", "LLM JUDGE", "CALIBRATION", "VERDICT"):
        assert header in rendered


def test_build_report_without_judge(ev: Any) -> None:
    texts = {"s:p1": "fix the crash", "s:p2": "design the cache"}
    report = ev.build_report(
        embedder=HashingEmbedder(seed=4),
        texts=texts,
        litmus=ev.load_litmus()[:5],
        judge_client=None,
    )
    assert report.judge is None
    assert "skipped" in ev.render_report(report)
