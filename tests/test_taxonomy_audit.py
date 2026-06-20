"""Tests B1.2 — taxonomy audit by clustering the whole corpus.

The whole path runs through :class:`HashingEmbedder` (deterministic, pure numpy)
so CI never downloads the real model. Within-group prompts share identical text
→ identical vectors → tight, well-separated clusters HDBSCAN finds reliably.
The diagnostic helpers (merge/split/theme detection) are also pinned directly on
hand-built clusters so their thresholds are tested without leaning on clustering.
"""

from __future__ import annotations

import csv
from pathlib import Path

from prompt_analytics import taxonomy_audit
from prompt_analytics.embeddings import HashingEmbedder
from prompt_analytics.schema import CATEGORIES_COLS, PROMPT_TEXT_COLS, PROMPTS_COLS
from prompt_analytics.taxonomy_audit import (
    Cluster,
    build_report,
    format_report,
    run_audit,
    write_audit_csv,
)

# ── synthetic corpus: 3 tight, disjoint semantic groups ───────────────────────

_GROUPS = {
    "implementation": "implement the new login endpoint feature",
    "debug": "fix the crash traceback error stacktrace",
    "review": "review audit the pull request changes",
}


def _corpus() -> tuple[list[str], list[str], dict[str, str]]:
    ids: list[str] = []
    texts: list[str] = []
    cats: dict[str, str] = {}
    for cat, text in _GROUPS.items():
        for i in range(6):
            pid = f"s1:{cat}{i}"
            ids.append(pid)
            texts.append(text)
            cats[pid] = cat
    return ids, texts, cats


def _report() -> taxonomy_audit.AuditReport:
    ids, texts, cats = _corpus()
    return build_report(
        HashingEmbedder(dim=256), ids, texts, cats, min_cluster_size=3
    )


# ── clustering + labelling ────────────────────────────────────────────────────


def test_clusters_the_three_groups() -> None:
    report = _report()
    assert report.n_prompts == 18
    assert report.n_clusters == 3
    assert report.noise == 0
    assert sum(c.size for c in report.clusters) == 18


def test_each_cluster_is_pure_and_labelled() -> None:
    report = _report()
    dominants = {c.dominant_category for c in report.clusters}
    assert dominants == {"implementation", "debug", "review"}
    for c in report.clusters:
        assert c.purity == 1.0  # disjoint groups → perfectly pure clusters
        # The label terms come from the group's own vocabulary (stopwords gone).
        group_words = set(_GROUPS[c.dominant_category].split())
        assert set(c.top_terms) & group_words
        assert c.representatives  # at least one archetype prompt


def test_categories_columns_in_taxonomy_order() -> None:
    report = _report()
    # debug precedes review precedes implementation in the canonical tie-break.
    assert report.categories == ["debug", "review", "implementation"]


# ── merge / split / theme detection (pinned on hand-built clusters) ───────────


def _cluster(cid: int, counts: dict[str, int], terms: list[str] | None = None) -> Cluster:
    size = sum(counts.values())
    dominant = max(counts, key=lambda k: counts[k])
    return Cluster(
        cluster_id=cid,
        size=size,
        category_counts=counts,
        dominant_category=dominant,
        purity=counts[dominant] / size,
        top_terms=terms or ["alpha", "beta"],
        representatives=["a prompt"],
    )


def test_detect_merges_flags_blended_clusters() -> None:
    # A cluster that is half test, half implementation → the classic merge.
    merges = taxonomy_audit._detect_merges(
        [_cluster(0, {"test": 5, "implementation": 5})]
    )
    assert len(merges) == 1
    assert set(merges[0].categories) == {"test", "implementation"}
    # A pure cluster, or one only diluted by "other", is not a merge.
    assert taxonomy_audit._detect_merges([_cluster(1, {"debug": 9, "other": 1})]) == []


def test_detect_splits_flags_scattered_category() -> None:
    # "refactor" spread evenly across four clusters → no dominant home.
    clusters = [_cluster(i, {"refactor": 4, "debug": 1}) for i in range(4)]
    splits = taxonomy_audit._detect_splits(clusters, ["refactor", "debug"])
    cats = {s.category for s in splits}
    assert "refactor" in cats
    refactor = next(s for s in splits if s.category == "refactor")
    assert len(refactor.cluster_ids) == 4
    assert refactor.top_share < taxonomy_audit._SPLIT_TOP_SHARE


def test_detect_splits_ignores_catch_all_buckets() -> None:
    clusters = [_cluster(i, {"other": 5}) for i in range(4)]
    assert taxonomy_audit._detect_splits(clusters, ["other"]) == []


def test_detect_themes_latent_other_and_cross_cutting() -> None:
    latent = _cluster(0, {"other": 8, "debug": 2})  # other-dominated, coherent
    mixed = _cluster(1, {"debug": 3, "test": 3, "plan": 2, "review": 2})  # no clear winner
    themes = {t.cluster_id: t.kind for t in taxonomy_audit._detect_themes([latent, mixed])}
    assert themes[0] == "latent-other"
    assert themes[1] == "cross-cutting"


def test_other_distribution_is_reported() -> None:
    ids, texts, cats = _corpus()
    # Relabel the whole debug group as "other": it should regroup into one cluster.
    for pid in list(cats):
        if pid.startswith("s1:debug"):
            cats[pid] = "other"
    report = build_report(HashingEmbedder(dim=256), ids, texts, cats, min_cluster_size=3)
    assert report.other_distribution
    assert report.other_distribution[0][1] == 6  # all six 'other' in one cluster


# ── run_audit end to end (I/O, no model download) ─────────────────────────────


def _write(path: Path, cols: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _setup(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    ids, texts, cats = _corpus()
    prompts = []
    text_rows = []
    cat_rows = []
    for pid, text in zip(ids, texts, strict=True):
        row = dict.fromkeys(PROMPTS_COLS, "")
        row["prompt_id"] = pid
        row["session_id"] = "s1"
        row["prompt_preview"] = text
        prompts.append(row)
        text_rows.append({"prompt_id": pid, "prompt_text": text})
        cat_rows.append(
            {
                "prompt_id": pid,
                "category": cats[pid],
                "complexity": "3",
                "classifier_model": "heuristic-v3",
                "classified_at": "x",
            }
        )
    _write(out / "prompts.csv", PROMPTS_COLS, prompts)
    _write(out / "prompts_text.csv", PROMPT_TEXT_COLS, text_rows)
    _write(out / "categories.csv", CATEGORIES_COLS, cat_rows)
    return out


def test_run_audit_writes_report_and_matrix(tmp_path: Path) -> None:
    out = _setup(tmp_path)
    code = run_audit(output_dir=str(out), embedder=HashingEmbedder(dim=256), min_cluster_size=3)
    assert code == 0
    assert (out / "taxonomy_audit.txt").exists()
    csv_path = out / "taxonomy_audit.csv"
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3  # three clusters
    # The alignment matrix carries one count column per present category.
    for cat in ("debug", "review", "implementation"):
        assert cat in rows[0]
    # Diagnostic: it never writes categories from the audit (pure diagnostic).
    assert {"taxonomy_audit.csv", "taxonomy_audit.txt"} <= {p.name for p in out.iterdir()}


def test_run_audit_no_prompts_returns_error(tmp_path: Path) -> None:
    out = tmp_path / "empty"
    out.mkdir()
    assert run_audit(output_dir=str(out), embedder=HashingEmbedder(dim=64)) == -1


def test_run_audit_no_text_returns_error(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    row = dict.fromkeys(PROMPTS_COLS, "")
    row["prompt_id"] = "s1:p1"
    row["session_id"] = "s1"
    _write(out / "prompts.csv", PROMPTS_COLS, [row])  # no preview, no text file
    assert run_audit(output_dir=str(out), embedder=HashingEmbedder(dim=64)) == -1


def test_run_audit_shares_embedding_cache(tmp_path: Path) -> None:
    out = _setup(tmp_path)
    run_audit(output_dir=str(out), embedder=HashingEmbedder(dim=256), min_cluster_size=3)
    assert (out / "embeddings.npz").exists()  # reuses the B1.0 disk cache


# ── report formatting + CSV hardening ─────────────────────────────────────────


def test_format_report_has_all_sections() -> None:
    text = "\n".join(format_report(_report()))
    for header in ("Merge signals", "Split signals", "Themes to eyeball", "'other' bucket"):
        assert header in text
    assert "pure diagnostic" in text  # the disclaimer is loud


def test_write_audit_csv_neutralizes_formula_injection(tmp_path: Path) -> None:
    cluster = Cluster(
        cluster_id=0,
        size=1,
        category_counts={"debug": 1},
        dominant_category="debug",
        purity=1.0,
        top_terms=["=cmd"],
        representatives=["=SUM(A1:A9)"],
    )
    report = taxonomy_audit.AuditReport(
        n_prompts=1, n_clusters=1, noise=0, categories=["debug"], clusters=[cluster]
    )
    path = tmp_path / "audit.csv"
    write_audit_csv(path, report)
    with path.open(encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["representatives"].startswith("'=")  # spreadsheet formula defused
    assert row["top_terms"].startswith("'=")
