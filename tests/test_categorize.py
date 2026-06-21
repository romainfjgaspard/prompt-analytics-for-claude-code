"""Tests 11.4 – categorize: heuristic, observed complexity, LLM mode."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prompt_analytics import categorize
from prompt_analytics.categorize import (
    _AnthropicBatchClassifier,
    _call_with_retry,
    _classify_heuristic,
    _observed_complexity_scores,
    _parse_reply,
    _PermanentError,
    _quantile_band,
    _TransientError,
    run_categorize,
)
from prompt_analytics.schema import CATEGORIES_COLS, PROMPTS_COLS

# ── fixtures / helpers ────────────────────────────────────────────────────────


def _base_row(prompt_id: str, **kw: str) -> dict[str, str]:
    row = dict.fromkeys(PROMPTS_COLS, "")
    row["prompt_id"] = prompt_id
    row["session_id"] = "s1"
    row.update(kw)
    return row


def _write_prompts(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=PROMPTS_COLS).writeheader()
        csv.DictWriter(fh, fieldnames=PROMPTS_COLS).writerows(rows)


def _write_categories(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CATEGORIES_COLS)
        w.writeheader()
        w.writerows(rows)


def _read_categories(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _make_out(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    return d


# ── heuristic: labeled FR/EN sample ──────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        # EN – debug
        ("fix the null pointer in user.py", "debug"),
        ("why is the test failing?", "debug"),
        ("the app is broken after the last commit", "debug"),
        # EN – implementation
        ("write a function to parse CSV files", "implementation"),
        ("add a login endpoint to the API", "implementation"),
        ("build the user registration form", "implementation"),
        # EN – refactor
        ("refactor the auth module to use the new token format", "refactor"),
        ("simplify the database query logic", "refactor"),
        ("rename the variable to be more descriptive", "refactor"),
        # EN – plan
        ("design an architecture for the distributed cache", "plan"),
        ("what is the best way to structure the new feature?", "plan"),
        ("how should we approach the migration strategy?", "plan"),
        # EN – question
        ("explain how the cache works", "question"),
        ("what does this function do?", "question"),
        ("tell me the difference between eager and lazy loading", "question"),
        # FR – debug
        ("corrige l'erreur dans le module auth", "debug"),
        ("pourquoi ça ne fonctionne pas après le merge?", "debug"),
        ("il y a un traceback dans les logs", "debug"),
        # FR – implementation
        ("implémente la fonction de parsing CSV", "implementation"),
        ("crée un endpoint de connexion", "implementation"),
        ("ajoute la validation des champs du formulaire", "implementation"),
        # FR – refactor
        ("refactorise le module auth", "refactor"),
        ("simplifie les requêtes SQL du service", "refactor"),
        ("renomme la variable pour plus de clarté", "refactor"),
        # FR – plan
        ("quelle architecture pour le cache distribué?", "plan"),
        ("comment structurer la nouvelle fonctionnalité?", "plan"),
        # FR – question
        ("explique comment fonctionne le cache", "question"),
        ("décris le fonctionnement de ce module", "question"),
        ("quelle est la différence entre les deux approches?", "question"),
    ],
)
def test_heuristic_labeled_sample(text: str, expected: str) -> None:
    assert _classify_heuristic(text) == expected, f"'{text}' → expected {expected}"


def test_heuristic_other_for_blank() -> None:
    assert _classify_heuristic("") == "other"
    assert _classify_heuristic("claude --help") == "other"
    assert _classify_heuristic("anglais") == "other"


@pytest.mark.parametrize(
    "text",
    [
        "ok",
        "yes",
        "oui",
        "non",
        "oui vas y",
        "ok go",
        "les deux",
        "reprend",
        "recommence",
        "ok c'est parti.",
        "1 et 3",
        "tu es bloqué?",
        "tu avais l'air bloqué depuis super longtemps...",
        "tu t'es encore figé",
        # Short option picks (extended ack vocabulary).
        "ok pour A",
        "ok pour l'option 1.",
        "oui partons sur l'etape 2",
    ],
)
def test_heuristic_followup_acks_and_nudges(text: str) -> None:
    assert _classify_heuristic(text) == "followup", f"'{text}' → expected followup"


def test_heuristic_pure_task_notification_is_notification() -> None:
    """A turn that is nothing but a harness task-notification gets its own bucket."""
    block = (
        "<task-notification><task-id>bh5</task-id>"
        "<status>completed</status>"
        "<summary>Background command finished</summary></task-notification>"
    )
    assert _classify_heuristic(block) == "notification"
    # A real instruction that merely follows a notification is classified on its
    # own words, not swallowed by the notification bucket.
    assert _classify_heuristic(block + " commit et push") == "ops"


def test_heuristic_ack_prefix_does_not_shortcircuit() -> None:
    """ "ok <real instruction>" must classify the instruction, not the "ok"."""
    assert _classify_heuristic("ok sauvegarde le fichier status") == "docs"
    assert _classify_heuristic("oui commit et push la branche") == "ops"


def test_heuristic_trailing_question_mark_fallback() -> None:
    """No rule fires → a prompt ending in "?" is a question, not "other"."""
    assert _classify_heuristic("il faut redemarrer pycharm?") == "question"
    assert _classify_heuristic("il faut redemarrer pycharm") == "other"


def test_heuristic_tie_break_is_deterministic() -> None:
    """Equal scores resolve by the fixed _TIE_BREAK_ORDER (debug first), not by
    set iteration order (which varies with the process hash seed)."""
    # "fix" (debug, 1.2) ties with "plan" (plan, 1.2).
    assert _classify_heuristic("fix the plan") == "debug"


@pytest.mark.parametrize(
    "text,expected",
    [
        # Conjugated/imperative FR forms that the v1 patterns missed.
        ("écris la doc de la fonction", "docs"),  # v2: doc work beats the verb
        ("génère un fichier de config", "implementation"),
        ("corrigez les imports cassés", "debug"),
        ("résous le conflit de merge", "debug"),
        ("nettoie le code mort du module", "refactor"),
        ("optimise la requête SQL", "refactor"),
        ("simplifie cette fonction", "refactor"),
        ("expliquez la différence entre les deux", "question"),
    ],
)
def test_heuristic_fr_conjugations(text: str, expected: str) -> None:
    assert _classify_heuristic(text) == expected, f"'{text}' → expected {expected}"


@pytest.mark.parametrize(
    "text,expected",
    [
        # v2 agentic categories (4.1), FR + EN, tuned on the real "other" set.
        ("fais un audit complet du projet, la qualité, etc", "review"),
        ("analyse le projet", "review"),
        ("vérifie la présence du fichier sur le sftp", "review"),
        ("review the changes before I merge", "review"),
        ("rajoute ces fichiers dans les tests integration", "test"),
        ("relance les tests d'intégration pour vérifier les régressions", "test"),
        ("add unit tests for the parser", "test"),
        ("met bien a jour les fichiers de status des deux projets", "docs"),
        ("update the README and the changelog", "docs"),
        ("commit bien tout sur une branche", "ops"),
        ("fais des merges de main sur ces deux branches, ensuite je ferais des PR", "ops"),
        ("commit and push the branch, then open a PR", "ops"),
        ("je vais lancer la pipeline de deploiement mais il faut que tu commit push", "ops"),
        # Unaccented real-world French the v1 rules missed (D3).
        ("ca a pas marché, j'ai supprimé une ligne", "debug"),
        ("les \\n dans le graphe ne marche pas", "debug"),
        ("il y a un probleme avec les erreurs distinctes", "debug"),
        ("en mode databricks 05 plante: XGBoostError: check failed", "debug"),
        ("comment je fais deja pour relancer ceux qui avaient des diffs?", "question"),
        ("c'est quoi la phase 4?", "question"),
        ("a quoi sert le premier bloc avec PROJECT_ROOT?", "question"),
        ("je ne comprend pas ton point sur les snapshots", "question"),
    ],
)
def test_heuristic_v2_agentic_categories(text: str, expected: str) -> None:
    assert _classify_heuristic(text) == expected, f"'{text}' → expected {expected}"


@pytest.mark.parametrize(
    "text,expected",
    [
        # v3 (4.x): the "feedback" category drains the biggest real-"other"
        # cluster -- mid-length course-correction / critique / preferences that
        # carry no concrete task keyword. FR (mostly unaccented) + EN.
        (
            "c'est pas mal mais l'ordre des modeles devrait etre du plus petit au plus gros",
            "feedback",
        ),
        ("en fait j'ai change d'avis, on part sur dark force", "feedback"),
        ("non plutot mets la legende a droite, elle chevauche les barres", "feedback"),
        ("ca change rien, c'est toujours pareil", "feedback"),
        ("je suis pas convaincu par cette approche", "feedback"),
        ("actually, I'd prefer the legend on the right", "feedback"),
        ("not quite, the order is still wrong", "feedback"),
    ],
)
def test_heuristic_feedback_category(text: str, expected: str) -> None:
    assert _classify_heuristic(text) == expected, f"'{text}' → expected {expected}"


@pytest.mark.parametrize(
    "text,expected",
    [
        # A concrete intent always out-scores a low-weight discourse marker, so
        # feedback never steals a real task prompt.
        ("en fait corrige le bug dans user.py", "debug"),
        ("par contre ajoute un test pour le parser", "test"),
        # Accent-tolerant stuck-assistant nudge stays followup over a feedback
        # marker ("a mon avis" + the unaccented "tu etais bloque").
        ("a mon avis tu etais bloque", "followup"),
    ],
)
def test_heuristic_feedback_loses_to_concrete_intent(text: str, expected: str) -> None:
    assert _classify_heuristic(text) == expected, f"'{text}' → expected {expected}"


def test_heuristic_strips_harness_chrome() -> None:
    """``<system-reminder>`` / ``<task-notification>`` wrappers are removed first."""
    # A pure system-reminder block carries no user intent -> other.
    assert _classify_heuristic("<system-reminder>just a note</system-reminder>") == "other"
    # The reminder prefix must not mask the real instruction after it.
    assert _classify_heuristic("<system-reminder>note</system-reminder> commit et push") == "ops"
    assert (
        _classify_heuristic("<system-reminder>sent at 10:00</system-reminder> oui c'est bon")
        == "followup"
    )


# ── _parse_reply edge cases ───────────────────────────────────────────────────


def test_parse_reply_normal() -> None:
    assert _parse_reply("debug|3") == ("debug", "3")


def test_parse_reply_uppercase_spaces() -> None:
    assert _parse_reply("  Implementation | 4 ") == ("implementation", "4")


def test_parse_reply_unknown_category() -> None:
    cat, comp = _parse_reply("typo|2")
    assert cat == "other"
    assert comp == "2"


def test_parse_reply_unknown_complexity() -> None:
    cat, comp = _parse_reply("debug|9")
    assert cat == "debug"
    assert comp == "3"  # fallback to median


def test_parse_reply_missing_pipe() -> None:
    cat, comp = _parse_reply("debug")
    assert cat == "debug"
    assert comp == "3"


def test_parse_reply_empty() -> None:
    assert _parse_reply("") == ("other", "3")
    assert _parse_reply(None) == ("other", "3")  # type: ignore[arg-type]


# ── observed complexity ───────────────────────────────────────────────────────


def test_quantile_band_quintiles() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _quantile_band(1.0, vals) == 1  # 20th percentile
    assert _quantile_band(2.0, vals) == 2  # 40th percentile
    assert _quantile_band(3.0, vals) == 3  # 60th percentile
    assert _quantile_band(4.0, vals) == 4  # 80th percentile
    assert _quantile_band(5.0, vals) == 5  # 100th percentile


def test_quantile_band_empty() -> None:
    assert _quantile_band(99.0, []) == 3


def test_observed_complexity_hand_calculated() -> None:
    """5 prompts with strictly increasing metrics → each lands in a clean quintile.

    Distribution for every dimension: [v1, v2, v3, v4, v5] strictly increasing.
    rank(vi) = i/5.  Quintile bands: 0.2→1, 0.4→2, 0.6→3, 0.8→4, 1.0→5.
    So p1 gets band 1 on all dims → avg 1.0 → "1", etc.
    """
    prompts: list[dict[str, Any]] = [
        {"prompt_id": "p1", "assistant_turns": "1", "tool_calls": "0", "char_count": "100"},
        {"prompt_id": "p2", "assistant_turns": "2", "tool_calls": "2", "char_count": "200"},
        {"prompt_id": "p3", "assistant_turns": "3", "tool_calls": "4", "char_count": "300"},
        {"prompt_id": "p4", "assistant_turns": "4", "tool_calls": "6", "char_count": "400"},
        {"prompt_id": "p5", "assistant_turns": "5", "tool_calls": "8", "char_count": "500"},
    ]
    # Costs also strictly increasing so cost dim matches others
    costs = {"p1": 0.001, "p2": 0.002, "p3": 0.003, "p4": 0.004, "p5": 0.005}
    scores = _observed_complexity_scores(prompts, costs)

    assert scores["p1"] == "1"  # rank=0.2 on all dims → band 1 → avg 1.0
    assert scores["p2"] == "2"  # rank=0.4 → band 2 → avg 2.0
    assert scores["p3"] == "3"  # rank=0.6 → band 3 → avg 3.0
    assert scores["p4"] == "4"  # rank=0.8 → band 4 → avg 4.0
    assert scores["p5"] == "5"  # rank=1.0 → band 5 → avg 5.0
    assert set(scores.keys()) == {"p1", "p2", "p3", "p4", "p5"}


def test_quantile_band_degenerate_all_equal() -> None:
    assert _quantile_band(0.0, [0.0, 0.0, 0.0]) == 3


def test_observed_complexity_empty() -> None:
    assert _observed_complexity_scores([], {}) == {}


# ── heuristic run: idempotent, no overwrite of LLM rows ──────────────────────


def test_heuristic_run_classifies_uncategorized(tmp_path: Path) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [
            _base_row("p1", prompt_preview="fix the null pointer in user.py"),
        ],
    )
    count = run_categorize(output_dir=str(out), delay=0)
    assert count == 1
    rows = _read_categories(out / "categories.csv")
    assert len(rows) == 1
    assert rows[0]["category"] == "debug"
    assert rows[0]["classifier_model"] == categorize.HEURISTIC_VERSION
    assert rows[0]["complexity"] in {"1", "2", "3", "4", "5"}


def test_heuristic_never_overwrites_llm_row(tmp_path: Path) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [
            _base_row("p1", prompt_preview="fix the null pointer"),
            _base_row("p2", prompt_preview="design the cache architecture"),
        ],
    )
    # p1 already classified by an LLM
    _write_categories(
        out / "categories.csv",
        [
            {
                "prompt_id": "p1",
                "category": "plan",  # deliberately "wrong" to prove no overwrite
                "complexity": "5",
                "classifier_model": "claude-haiku-4-5",
                "classified_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    )
    count = run_categorize(output_dir=str(out), delay=0)
    assert count == 1  # only p2 was classified
    cats = {r["prompt_id"]: r for r in _read_categories(out / "categories.csv")}
    assert cats["p1"]["category"] == "plan"  # unchanged
    assert cats["p1"]["classifier_model"] == "claude-haiku-4-5"  # unchanged
    assert cats["p2"]["classifier_model"] == categorize.HEURISTIC_VERSION


def test_missing_prompts_file_returns_minus_one(tmp_path: Path) -> None:
    """ "Could not attempt" (-1, CLI exits 1) differs from "nothing new" (0)."""
    assert run_categorize(output_dir=str(tmp_path / "nope"), delay=0) == -1


def test_llm_without_any_key_returns_minus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _make_out(tmp_path)
    _write_prompts(out / "prompts.csv", [_base_row("p1", prompt_preview="fix the bug")])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    assert run_categorize(output_dir=str(out), use_llm=True, delay=0) == -1


def test_heuristic_upgrades_older_heuristic_rows_not_llm(tmp_path: Path) -> None:
    """A rules upgrade re-classifies heuristic-vN rows (4.2) but never LLM rows."""
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [
            _base_row("p1", prompt_preview="commit bien tout sur une branche"),
            _base_row("p2", prompt_preview="commit bien tout sur une branche"),
        ],
    )
    _write_categories(
        out / "categories.csv",
        [
            {
                "prompt_id": "p1",
                "category": "other",  # what heuristic-v1 said
                "complexity": "2",
                "classifier_model": "heuristic-v1",
                "classified_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "prompt_id": "p2",
                "category": "other",  # LLM said so: must survive the upgrade
                "complexity": "2",
                "classifier_model": "claude-haiku-4-5",
                "classified_at": "2026-01-01T00:00:00+00:00",
            },
        ],
    )
    count = run_categorize(output_dir=str(out), delay=0)
    assert count == 1  # only the stale heuristic row
    cats = {r["prompt_id"]: r for r in _read_categories(out / "categories.csv")}
    assert cats["p1"]["category"] == "ops"
    assert cats["p1"]["classifier_model"] == categorize.HEURISTIC_VERSION
    assert cats["p2"]["category"] == "other"
    assert cats["p2"]["classifier_model"] == "claude-haiku-4-5"


def test_llm_mode_does_not_reclassify_stale_heuristic_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heuristic version bump only drives *heuristic* runs; --llm fills
    blanks and never re-spends API calls on already-categorized rows."""
    out = _make_out(tmp_path)
    _write_prompts(out / "prompts.csv", [_base_row("p1", prompt_preview="fix the bug")])
    _write_categories(
        out / "categories.csv",
        [
            {
                "prompt_id": "p1",
                "category": "debug",
                "complexity": "2",
                "classifier_model": "heuristic-v1",
                "classified_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    )
    fake = _FakeLLM([])
    monkeypatch.setattr(categorize, "build_client", lambda **_kw: fake)
    count = run_categorize(output_dir=str(out), use_llm=True, delay=0)
    assert count == 0
    assert fake.calls == []


def test_heuristic_idempotent(tmp_path: Path) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [
            _base_row("p1", prompt_preview="refactor the auth module"),
        ],
    )
    count1 = run_categorize(output_dir=str(out), delay=0)
    count2 = run_categorize(output_dir=str(out), delay=0)
    assert count1 == 1
    assert count2 == 0  # nothing new to classify


def test_pseudo_prompts_skipped(tmp_path: Path) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [
            _base_row("s1:_continuation", prompt_preview="continuation text"),
            _base_row("p1", prompt_preview="fix the bug"),
        ],
    )
    count = run_categorize(output_dir=str(out), delay=0)
    assert count == 1
    cats = {r["prompt_id"]: r for r in _read_categories(out / "categories.csv")}
    assert "s1:_continuation" not in cats
    assert "p1" in cats


def test_no_text_extract_leaves_category_empty_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After `extract --no-text` (no prompts_text.csv, blank previews) nothing
    must be filed under "other" -- skip loudly and leave the category empty so
    a later text-enabled extract + categorize classifies for real (N2)."""
    out = _make_out(tmp_path)
    _write_prompts(out / "prompts.csv", [_base_row("p1"), _base_row("p2")])
    count = run_categorize(output_dir=str(out), delay=0)
    assert count == 0
    err = capsys.readouterr().err
    assert "no stored text" in err and "--no-text" in err
    rows = _read_categories(out / "categories.csv")
    assert len(rows) == 2  # observed complexity is still computed
    assert all(r["category"] == "" for r in rows)
    assert all(r["classifier_model"] == "" for r in rows)
    assert all(r["complexity"] in {"1", "2", "3", "4", "5"} for r in rows)


def test_no_text_prompts_skipped_but_textful_ones_classified(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [_base_row("p1"), _base_row("p2", prompt_preview="fix the bug")],
    )
    count = run_categorize(output_dir=str(out), delay=0)
    assert count == 1
    assert "1 prompt(s) have no stored text" in capsys.readouterr().err
    cats = {r["prompt_id"]: r for r in _read_categories(out / "categories.csv")}
    assert cats["p1"]["category"] == ""  # skipped, retried on a later run
    assert cats["p2"]["category"] == "debug"


def test_limit_caps_classifications(tmp_path: Path) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv", [_base_row(f"p{i}", prompt_preview=f"fix bug {i}") for i in range(5)]
    )
    count = run_categorize(output_dir=str(out), delay=0, limit=2)
    assert count == 2
    assert len(_read_categories(out / "categories.csv")) == 5  # all get complexity; 2 get category
    categorized = [r for r in _read_categories(out / "categories.csv") if r["category"]]
    assert len(categorized) == 2


def test_complexity_updated_for_existing_llm_row(tmp_path: Path) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [
            _base_row("p1", prompt_preview="fix bug", assistant_turns="5", tool_calls="10"),
            _base_row("p2", prompt_preview="ok", assistant_turns="0", tool_calls="0"),
        ],
    )
    _write_categories(
        out / "categories.csv",
        [
            {
                "prompt_id": "p1",
                "category": "debug",
                "complexity": "1",  # stale LLM-guessed complexity
                "classifier_model": "claude-haiku-4-5",
                "classified_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    )
    run_categorize(output_dir=str(out), delay=0)
    cats = {r["prompt_id"]: r for r in _read_categories(out / "categories.csv")}
    # p1 has high metrics → should be bumped above "1"
    assert int(cats["p1"]["complexity"]) > 1
    # category preserved
    assert cats["p1"]["category"] == "debug"


# ── LLM serial mode: retry / permanent / no data loss ────────────────────────


class _FakeLLM:
    model = "fake-llm"
    calls: list[str]
    results: list[tuple[str, str] | Exception]

    def __init__(self, results: list[tuple[str, str] | Exception]) -> None:
        self.results = list(results)
        self.calls = []

    def classify(self, text: str) -> tuple[str, str]:
        self.calls.append(text)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_llm_transient_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fn() -> tuple[str, str]:
        calls.append(1)
        if len(calls) < 3:
            raise _TransientError("flaky")
        return ("debug", "2")

    result = _call_with_retry(fn, max_retries=3, base_delay=0)
    assert result == ("debug", "2")
    assert len(calls) == 3


def test_llm_permanent_error_raises() -> None:
    def fn() -> tuple[str, str]:
        raise _PermanentError("invalid key")

    with pytest.raises(_PermanentError, match="invalid key"):
        _call_with_retry(fn, max_retries=3, base_delay=0)


def test_llm_exhausted_retries_returns_none() -> None:
    def fn() -> tuple[str, str]:
        raise _TransientError("always fails")

    result = _call_with_retry(fn, max_retries=2, base_delay=0)
    assert result is None


def test_llm_permanent_aborts_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [
            _base_row("p1", prompt_preview="first"),
            _base_row("p2", prompt_preview="second"),
            _base_row("p3", prompt_preview="third"),
        ],
    )
    fake = _FakeLLM(
        [
            ("debug", "2"),
            _PermanentError("invalid API key"),
            ("plan", "4"),
        ]
    )
    monkeypatch.setattr(categorize, "build_client", lambda **_kw: fake)
    count = run_categorize(output_dir=str(out), use_llm=True, delay=0)
    # aborted after p2 raises; p1 was already saved
    assert count == 1
    cats = {r["prompt_id"]: r for r in _read_categories(out / "categories.csv")}
    assert cats["p1"]["category"] == "debug"
    assert cats.get("p2", {}).get("category", "") == ""


def test_llm_checkpoint_no_data_loss_on_mid_batch_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a crash after batch 1 completes: batch 0 data must survive."""
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv", [_base_row(f"p{i}", prompt_preview=f"prompt {i}") for i in range(6)]
    )

    call_count = [0]

    class CrashingLLM:
        model = "crashing"

        def classify(self, text: str) -> tuple[str, str]:
            call_count[0] += 1
            if call_count[0] > 3:
                raise RuntimeError("simulated crash")
            return ("debug", "2")

    monkeypatch.setattr(categorize, "build_client", lambda **_kw: CrashingLLM())

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_categorize(output_dir=str(out), use_llm=True, delay=0, batch_size=3)

    # The first 3 were checkpointed before the crash
    cats = _read_categories(out / "categories.csv")
    classified = [r for r in cats if r["category"] == "debug"]
    assert len(classified) == 3


def test_llm_delay_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv",
        [
            _base_row("p1", prompt_preview="fix it"),
            _base_row("p2", prompt_preview="add it"),
        ],
    )
    fake = _FakeLLM([("debug", "2"), ("implementation", "3")])
    monkeypatch.setattr(categorize, "build_client", lambda **_kw: fake)

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    run_categorize(output_dir=str(out), use_llm=True, delay=0.05)
    assert any(s == pytest.approx(0.05) for s in sleep_calls)


# ── LLM batch mode ────────────────────────────────────────────────────────────


def test_llm_batch_mode_uses_classify_many(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _make_out(tmp_path)
    _write_prompts(
        out / "prompts.csv", [_base_row(f"p{i}", prompt_preview=f"prompt {i}") for i in range(4)]
    )

    class FakeBatch(_AnthropicBatchClassifier):
        called_with: list[list[tuple[str, str]]] = []

        def __init__(self) -> None:
            self.model = "fake-batch"
            self._client = None  # type: ignore[assignment]

        def classify_many(self, items: list[tuple[str, str]]) -> dict[str, tuple[str, str]]:
            FakeBatch.called_with.append(items)
            return {pid: ("debug", "2") for pid, _ in items}

    fake = FakeBatch()
    monkeypatch.setattr(categorize, "build_client", lambda **_kw: fake)
    monkeypatch.setattr(categorize, "_AnthropicBatchClassifier", FakeBatch)

    count = run_categorize(output_dir=str(out), use_llm=True, use_batch=True, batch_size=2, delay=0)
    assert count == 4
    # Two batches of 2
    assert len(FakeBatch.called_with) == 2
    assert all(len(b) == 2 for b in FakeBatch.called_with)


# ── provider classifier wrappers (fake clients, no network) ────────────────────


def _ns(**kw: Any) -> SimpleNamespace:
    return SimpleNamespace(**kw)


def _chat_reply(content: str | None) -> SimpleNamespace:
    """Mimic an OpenAI-style chat.completions.create response."""
    return _ns(choices=[_ns(message=_ns(content=content))])


def test_anthropic_classifier_success() -> None:
    from prompt_analytics.categorize import _AnthropicClassifier

    class FakeMessages:
        def create(self, **_kw: Any) -> SimpleNamespace:
            return _ns(content=[_ns(text="debug | 3")])

    clf = _AnthropicClassifier(_ns(messages=FakeMessages()), model="m")  # type: ignore[arg-type]
    assert clf.classify("fix the crash") == ("debug", "3")


def test_anthropic_batch_classify_many_filters_failures() -> None:
    from prompt_analytics.categorize import _AnthropicBatchClassifier

    class FakeBatches:
        def create(self, requests: Any) -> SimpleNamespace:
            self.requests = list(requests)
            return _ns(id="batch_1")

        def retrieve(self, _bid: str) -> SimpleNamespace:
            return _ns(processing_status="ended")

        def results(self, _bid: str) -> list[SimpleNamespace]:
            return [
                _ns(
                    custom_id="p1",
                    result=_ns(
                        type="succeeded",
                        message=_ns(content=[_ns(text="plan | 4")]),
                    ),
                ),
                _ns(custom_id="p2", result=_ns(type="errored", message=None)),
            ]

    client = _ns(messages=_ns(batches=FakeBatches()))
    clf = _AnthropicBatchClassifier(client, model="m")  # type: ignore[arg-type]
    out = clf.classify_many([("p1", "design it"), ("p2", "broken")])
    assert out == {"p1": ("plan", "4")}  # the errored result is dropped


def test_anthropic_batch_poll_bounded_by_timeout(monkeypatch: Any) -> None:
    """3.5: a batch that never ends raises instead of looping forever."""
    from prompt_analytics.categorize import _AnthropicBatchClassifier, _PermanentError

    class NeverEnds:
        def create(self, requests: Any) -> SimpleNamespace:
            return _ns(id="batch_stuck")

        def retrieve(self, _bid: str) -> SimpleNamespace:
            return _ns(processing_status="in_progress")

    monkeypatch.setattr(categorize, "BATCH_POLL_TIMEOUT", 0)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = _ns(messages=_ns(batches=NeverEnds()))
    clf = _AnthropicBatchClassifier(client, model="m")  # type: ignore[arg-type]
    with pytest.raises(_PermanentError, match="giving up"):
        clf.classify_many([("p1", "x")])


def test_load_prompt_costs_does_not_swallow_pricing_error(tmp_path: Path, monkeypatch: Any) -> None:
    """3.6: a corrupt pricing.yml surfaces instead of silently zero-costing."""
    from prompt_analytics import analytics
    from prompt_analytics.categorize import _load_prompt_costs
    from prompt_analytics.pricing import PricingError

    tokens = tmp_path / "tokens.csv"
    tokens.write_text(
        "session_id,prompt_id,model,token_type,is_sidechain,token_count\n"
        "s1,p1,claude-opus-4-8,input,0,1000\n",
        encoding="utf-8",
    )

    def boom(*_a: Any, **_k: Any) -> None:
        raise PricingError("broken grid")

    monkeypatch.setattr(analytics, "get_model_pricing", boom)
    with pytest.raises(PricingError, match="broken grid"):
        _load_prompt_costs(tokens)


def test_load_prompt_costs_missing_file_is_empty(tmp_path: Path) -> None:
    from prompt_analytics.categorize import _load_prompt_costs

    assert _load_prompt_costs(tmp_path / "nope.csv") == {}


def test_openrouter_classifier_success_and_error_mapping() -> None:
    from prompt_analytics.categorize import (
        _OpenRouterClassifier,
        _PermanentError,
        _TransientError,
    )

    class FakeChat:
        def __init__(self, reply: str | None = None, exc: Exception | None = None) -> None:
            self._reply, self._exc = reply, exc

        @property
        def chat(self) -> SimpleNamespace:
            def create(**_kw: Any) -> SimpleNamespace:
                if self._exc:
                    raise self._exc
                return _chat_reply(self._reply)

            return _ns(completions=_ns(create=create))

    ok = _OpenRouterClassifier(FakeChat(reply="refactor | 2"), model="m")  # type: ignore[arg-type]
    assert ok.classify("clean it up") == ("refactor", "2")

    # Error mapping is by exception class name (the SDK is duck-typed here).
    auth_exc = type("AuthenticationError", (Exception,), {})()
    rate_exc = type("RateLimitError", (Exception,), {})()
    with pytest.raises(_PermanentError):
        _OpenRouterClassifier(FakeChat(exc=auth_exc))._call("x")  # type: ignore[arg-type]
    with pytest.raises(_TransientError):
        _OpenRouterClassifier(FakeChat(exc=rate_exc))._call("x")  # type: ignore[arg-type]


def test_azure_classifier_success_and_token_param() -> None:
    """Azure must request ``max_completion_tokens`` (reasoning models reject the
    16-token ``max_tokens`` cap and would return an empty reply)."""
    from prompt_analytics.categorize import (
        AZURE_MAX_COMPLETION_TOKENS,
        _AzureOpenAIClassifier,
        _PermanentError,
        _TransientError,
    )

    captured: dict[str, Any] = {}

    class FakeChat:
        def __init__(self, reply: str | None = None, exc: Exception | None = None) -> None:
            self._reply, self._exc = reply, exc

        @property
        def chat(self) -> SimpleNamespace:
            def create(**kw: Any) -> SimpleNamespace:
                captured.update(kw)
                if self._exc:
                    raise self._exc
                return _chat_reply(self._reply)

            return _ns(completions=_ns(create=create))

    ok = _AzureOpenAIClassifier(FakeChat(reply="implementation | 3"), model="dep")  # type: ignore[arg-type]
    assert ok.classify("build the endpoint") == ("implementation", "3")
    assert captured.get("max_completion_tokens") == AZURE_MAX_COMPLETION_TOKENS
    assert "max_tokens" not in captured  # reasoning models reject it

    auth_exc = type("AuthenticationError", (Exception,), {})()
    rate_exc = type("RateLimitError", (Exception,), {})()
    with pytest.raises(_PermanentError):
        _AzureOpenAIClassifier(FakeChat(exc=auth_exc), model="d")._call("x")  # type: ignore[arg-type]
    with pytest.raises(_TransientError):
        _AzureOpenAIClassifier(FakeChat(exc=rate_exc), model="d")._call("x")  # type: ignore[arg-type]


def test_build_client_azure_explicit_and_auto(
    _no_dotenv: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prompt_analytics.categorize import _AzureOpenAIClassifier

    # Explicit provider with an incomplete env → None (needs key AND endpoint).
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-test")
    assert categorize.build_client(provider="azure") is None
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-judge")
    client = categorize.build_client(provider="azure")
    assert isinstance(client, _AzureOpenAIClassifier)
    assert client.model == "gpt-judge"
    # auto with only Azure keys present falls through to Azure.
    assert isinstance(categorize.build_client(provider="auto"), _AzureOpenAIClassifier)


def test_ollama_classifier_success_and_error_is_blank() -> None:
    from prompt_analytics.categorize import _OllamaClassifier

    def make(reply: str | None = None, exc: Exception | None = None) -> SimpleNamespace:
        def create(**_kw: Any) -> SimpleNamespace:
            if exc:
                raise exc
            return _chat_reply(reply)

        return _ns(chat=_ns(completions=_ns(create=create)))

    ok = _OllamaClassifier(make(reply="question | 1"), model="m")  # type: ignore[arg-type]
    assert ok.classify("what is this") == ("question", "1")

    # A local Ollama outage must not raise — it leaves the row blank.
    blank = _OllamaClassifier(make(exc=RuntimeError("connection refused")))  # type: ignore[arg-type]
    assert blank.classify("anything") == ("", "")


# ── build_client provider selection ───────────────────────────────────────────


@pytest.fixture
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize .env loading and clear provider keys for deterministic tests."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)


def test_build_client_no_key_returns_none(_no_dotenv: None) -> None:
    assert categorize.build_client(provider="auto") is None
    assert categorize.build_client(provider="anthropic") is None
    assert categorize.build_client(provider="openrouter") is None


def test_build_client_anthropic_and_batch(
    _no_dotenv: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prompt_analytics.categorize import _AnthropicBatchClassifier, _AnthropicClassifier

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    single = categorize.build_client(provider="anthropic")
    assert isinstance(single, _AnthropicClassifier)
    batch = categorize.build_client(provider="anthropic", use_batch=True)
    assert isinstance(batch, _AnthropicBatchClassifier)


def test_build_client_openrouter_and_ollama(
    _no_dotenv: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prompt_analytics.categorize import _OllamaClassifier, _OpenRouterClassifier

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    assert isinstance(categorize.build_client(provider="openrouter"), _OpenRouterClassifier)
    # auto with only an OpenRouter key falls through to OpenRouter.
    assert isinstance(categorize.build_client(provider="auto"), _OpenRouterClassifier)
    # Ollama needs no key at all.
    assert isinstance(categorize.build_client(provider="ollama"), _OllamaClassifier)
