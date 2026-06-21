"""Tests B1.1 — offline semantic classifier (mono-label).

The whole scoring path is exercised through :class:`HashingEmbedder` (the
deterministic pure-numpy double), so the suite never downloads the real
``model2vec`` model. The hashing double is *not* semantically accurate (vectors
are random per token); these tests pin the **logic** — multi-prototype scoring,
the τ threshold, the fused lexical prime, deterministic tie-break, the hard
short-circuits, and the run_categorize integration — using texts whose token
overlap makes the cosines predictable.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from prompt_analytics import categorize
from prompt_analytics.categorize import (
    SEMANTIC_VERSION,
    SemanticClassifier,
    build_system_prompt,
    load_anchors,
    run_categorize,
)
from prompt_analytics.embeddings import HashingEmbedder
from prompt_analytics.schema import CATEGORIES_COLS, PROMPT_TEXT_COLS, PROMPTS_COLS

# ── shared anchors / SYSTEM_PROMPT (single source of truth) ───────────────────


def test_bundled_anchors_load_and_validate() -> None:
    anchors = load_anchors()
    # Every category in the file is a real taxonomy label with a valid role.
    assert set(anchors) <= categorize.CATEGORIES
    for name, spec in anchors.items():
        assert spec["role"] in categorize._VALID_ROLES, name
        assert spec["definition"]
    # Semantic categories must carry prototypes; "other" must not.
    assert anchors["other"]["role"] == "fallback"
    semantic = [n for n, s in anchors.items() if s["role"] == "semantic"]
    assert semantic and all(anchors[n].get("examples") for n in semantic)


def test_system_prompt_is_rendered_from_anchors() -> None:
    anchors = load_anchors()
    # The shared definitions feed the LLM prompt → the two modes can't drift.
    for name, spec in anchors.items():
        assert f"- {name}: " in categorize.SYSTEM_PROMPT
        assert " ".join(str(spec["definition"]).split())[:20] in categorize.SYSTEM_PROMPT


def test_build_system_prompt_from_custom_anchors() -> None:
    custom = {
        "plan": {"definition": "do the planning thing", "role": "semantic", "examples": ["x"]}
    }
    prompt = build_system_prompt(custom)
    assert "- plan: do the planning thing" in prompt
    assert "Complexity (1-5):" in prompt  # the curated tail is still appended


@pytest.mark.parametrize(
    "bad",
    [
        {"categories": {}},  # empty
        {"categories": {"plan": {"definition": "d", "role": "bogus", "examples": ["x"]}}},
        {"categories": {"nope": {"definition": "d", "role": "semantic", "examples": ["x"]}}},
        {"categories": {"plan": {"definition": "d", "role": "semantic"}}},  # no examples
        {"categories": {"plan": {"role": "semantic", "examples": ["x"]}}},  # no definition
    ],
)
def test_load_anchors_rejects_malformed(tmp_path: Path, bad: dict[str, Any]) -> None:
    import yaml

    p = tmp_path / "anchors.yml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_anchors(p)


# ── multi-prototype scoring (similarity max / top-k) ──────────────────────────

# Two semantic categories, distinct prototypes; ops as a lexical-prime category.
_ANCHORS = {
    "debug": {"definition": "d", "role": "semantic", "examples": ["fix the bug"]},
    "plan": {"definition": "p", "role": "semantic", "examples": ["design the architecture"]},
    "ops": {"definition": "o", "role": "lexical", "examples": []},
}


def _clf(**kw: object) -> SemanticClassifier:
    return SemanticClassifier(HashingEmbedder(dim=256), anchors=_ANCHORS, **kw)  # type: ignore[arg-type]


def test_prototype_match_picks_its_category() -> None:
    clf = _clf()
    # Text identical to a prototype → cosine 1.0 to it, near-0 to the other.
    assert clf.classify("fix the bug") == "debug"
    assert clf.classify("design the architecture") == "plan"


def test_top_k_mean_dilutes_a_single_strong_prototype() -> None:
    anchors = {
        "debug": {
            "role": "semantic",
            "definition": "d",
            "examples": ["alpha beta", "gamma delta"],  # disjoint vocab
        }
    }
    emb = HashingEmbedder(dim=256)
    by_max = SemanticClassifier(emb, anchors=anchors, top_k=1)
    by_mean = SemanticClassifier(emb, anchors=anchors, top_k=2)
    prep = by_max.prepare("alpha beta")
    vec = emb.embed(["alpha beta"])[0]
    s_max = by_max._scores("alpha beta", vec)["debug"]
    s_mean = by_mean._scores("alpha beta", vec)["debug"]
    assert s_max == pytest.approx(1.0, abs=0.05)  # max over prototypes
    assert s_mean < s_max  # the unrelated second prototype drags the mean down
    assert by_max.label(prep, vec) == "debug"


# ── τ threshold → other ───────────────────────────────────────────────────────


def test_below_threshold_is_other() -> None:
    clf = _clf(tau=0.30)
    # Vocabulary shared with no prototype and no lexical hit → all scores ~0.
    assert clf.classify("zzz qqq wxyz") == "other"


def test_threshold_governs_other_volume() -> None:
    # A high τ rejects an otherwise-winning semantic match; a low τ accepts it.
    emb = HashingEmbedder(dim=256)
    strict = SemanticClassifier(emb, anchors=_ANCHORS, tau=0.99)
    loose = SemanticClassifier(emb, anchors=_ANCHORS, tau=0.10)
    # Partial overlap with the "fix the bug" prototype: a middling cosine.
    text = "please fix the broken thing"
    assert loose.classify(text) == "debug"
    assert strict.classify(text) == "other"


# ── fused lexical prime (ops / feedback compete, never override) ───────────────


def test_lexical_prime_wins_when_no_semantic_match() -> None:
    clf = _clf()
    # Pure git ops, unrelated to any prototype → ops via the lexical prime.
    assert clf.classify("commit and push the branch") == "ops"


def test_lexical_prime_competes_but_loses_to_stronger_intent() -> None:
    clf = _clf()
    text = "fix the bug then commit it"
    scores = clf._scores(clf.prepare(text).text, clf.embedder.embed([text])[0])
    assert scores["ops"] > 0.0  # the prime is present and competing...
    assert scores["debug"] > scores["ops"]  # ...but the dominant intent wins
    assert clf.classify(text) == "debug"


def test_prime_weight_scales_the_lexical_contribution() -> None:
    text = "commit and push"
    light = _clf(prime_weight=0.0)
    heavy = _clf(prime_weight=0.70)
    assert light._scores(text, light.embedder.embed([text])[0]).get("ops", 0.0) == 0.0
    assert heavy._scores(text, heavy.embedder.embed([text])[0])["ops"] > 0.0


def test_feedback_lexical_prime_fires() -> None:
    anchors = {
        "debug": {"definition": "d", "role": "semantic", "examples": ["fix the bug"]},
        "feedback": {"definition": "f", "role": "lexical", "examples": []},
    }
    clf = SemanticClassifier(HashingEmbedder(dim=256), anchors=anchors)
    assert clf.classify("en fait, je prefere l'autre approche") == "feedback"


# ── deterministic tie-break ────────────────────────────────────────────────────


def test_tie_break_is_deterministic_and_specific_first() -> None:
    # Two semantic categories sharing the *same* prototype → identical cosine.
    anchors = {
        "plan": {"definition": "p", "role": "semantic", "examples": ["alpha beta gamma"]},
        "debug": {"definition": "d", "role": "semantic", "examples": ["alpha beta gamma"]},
    }
    # debug precedes plan in _TIE_BREAK_ORDER → it must win the tie, every run.
    assert categorize._TIE_BREAK_ORDER.index("debug") < categorize._TIE_BREAK_ORDER.index("plan")
    results = {
        SemanticClassifier(HashingEmbedder(dim=128), anchors=anchors).classify("alpha beta gamma")
        for _ in range(5)
    }
    assert results == {"debug"}


# ── hard short-circuits ────────────────────────────────────────────────────────


def test_task_notification_block_short_circuits() -> None:
    clf = _clf()
    text = "<task-notification>background task finished</task-notification>"
    assert clf.classify(text) == "notification"


def test_short_acknowledgement_is_followup() -> None:
    clf = _clf()
    assert clf.classify("oui, vas-y") == "followup"
    assert clf.classify("ok go") == "followup"
    # A real instruction after an ack is NOT a followup (the prime/semantic runs).
    assert clf.classify("ok, commit and push the branch") == "ops"


# ── run_categorize integration (mode plumbing, no model download) ─────────────


def _write(path: Path, cols: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _prompt_row(pid: str) -> dict[str, str]:
    row = dict.fromkeys(PROMPTS_COLS, "")
    row["prompt_id"] = pid
    row["session_id"] = "s1"
    return row


def _setup(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    _write(out / "prompts.csv", PROMPTS_COLS, [_prompt_row("s1:p1"), _prompt_row("s1:p2")])
    _write(
        out / "prompts_text.csv",
        PROMPT_TEXT_COLS,
        [
            {"prompt_id": "s1:p1", "prompt_text": "commit and push the branch"},
            {"prompt_id": "s1:p2", "prompt_text": "oui"},
        ],
    )
    return out


def _categories(out: Path) -> dict[str, dict[str, str]]:
    with (out / "categories.csv").open(encoding="utf-8", newline="") as fh:
        return {r["prompt_id"]: r for r in csv.DictReader(fh)}


def test_run_categorize_semantic_writes_stamped_categories(tmp_path: Path) -> None:
    out = _setup(tmp_path)
    n = run_categorize(output_dir=str(out), use_semantic=True, embedder=HashingEmbedder(dim=256))
    assert n == 2
    rows = _categories(out)
    assert rows["s1:p1"]["category"] == "ops"
    assert rows["s1:p2"]["category"] == "followup"
    assert all(r["classifier_model"] == SEMANTIC_VERSION for r in rows.values())
    assert (out / "embeddings.npz").exists()  # the disk cache (B1.0) is used


def test_run_categorize_semantic_is_idempotent(tmp_path: Path) -> None:
    out = _setup(tmp_path)
    run_categorize(output_dir=str(out), use_semantic=True, embedder=HashingEmbedder(dim=256))
    again = run_categorize(
        output_dir=str(out), use_semantic=True, embedder=HashingEmbedder(dim=256)
    )
    assert again == 0  # current version → nothing to redo


def test_semantic_supersedes_heuristic_but_not_llm(tmp_path: Path) -> None:
    out = _setup(tmp_path)
    _write(
        out / "categories.csv",
        CATEGORIES_COLS,
        [
            {
                "prompt_id": "s1:p1",
                "category": "question",
                "complexity": "3",
                "classifier_model": "heuristic-v3",
                "classified_at": "x",
            },
            {
                "prompt_id": "s1:p2",
                "category": "review",
                "complexity": "3",
                "classifier_model": "claude-haiku-4-5",
                "classified_at": "x",
            },
        ],
    )
    run_categorize(output_dir=str(out), use_semantic=True, embedder=HashingEmbedder(dim=256))
    rows = _categories(out)
    # heuristic row was superseded by semantic...
    assert rows["s1:p1"]["category"] == "ops"
    assert rows["s1:p1"]["classifier_model"] == SEMANTIC_VERSION
    # ...the LLM row was left untouched (more authoritative).
    assert rows["s1:p2"]["category"] == "review"
    assert rows["s1:p2"]["classifier_model"] == "claude-haiku-4-5"


def test_config_overrides_tau(tmp_path: Path) -> None:
    out = _setup(tmp_path)
    (out / "config.yml").write_text("semantic:\n  tau: 0.99\n", encoding="utf-8")
    run_categorize(output_dir=str(out), use_semantic=True, embedder=HashingEmbedder(dim=256))
    rows = _categories(out)
    # τ=0.99 rejects the ops prime (0.70-scaled) → falls through to other.
    assert rows["s1:p1"]["category"] == "other"
