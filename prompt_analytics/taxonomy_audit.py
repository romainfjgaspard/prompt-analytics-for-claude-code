"""Taxonomy audit by clustering the whole corpus (Axe B1.2).

``categorize --audit-categories`` is a **pure diagnostic**: it embeds *every*
prompt, clusters the vectors with HDBSCAN (density-based — no ``k`` to guess,
clusters of any shape/size, noise isolated), then compares the natural clusters
to our 13 hand-written categories. It answers "where does the data's own
structure agree with our taxonomy, and where does it diverge?":

* an **alignment matrix** (cluster x category counts);
* each cluster **labelled** by its distinctive terms (c-TF-IDF) and a few
  representative prompts (closest to the cluster centroid);
* **merge** signals — two categories that collapse into one natural cluster
  (the historical ``test`` / ``implementation`` complaint);
* **split** signals — one category scattered across several clusters;
* **cross-cutting** themes — mixed clusters that cut across our labels;
* **latent** themes — coherent clusters currently dumped in ``other`` (candidate
  new categories), plus how the ``other`` bucket regroups.

It **never writes ``categories.csv`` nor changes the taxonomy**: unsupervised
clusters are unstable and hard to name, so they *inform* a deliberate human
revision, they do not perform one. The output is a report (stdout + a ``.txt``)
and an alignment-matrix CSV. Tests drive the whole path through the
deterministic :class:`~prompt_analytics.embeddings.HashingEmbedder`, so CI never
downloads the real model.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .categorize import (
    _NOISE_WRAPPER_RE,
    _is_pseudo,
    _load_categories,
    _load_texts,
)
from .storage import atomic_write_csv, escape_csv_formula

if TYPE_CHECKING:
    from .embeddings import Embedder, FloatMatrix

__all__ = [
    "Cluster",
    "MergeSignal",
    "SplitSignal",
    "ThemeSignal",
    "AuditReport",
    "build_report",
    "format_report",
    "write_audit_csv",
    "run_audit",
    "DEFAULT_MIN_CLUSTER_SIZE",
]

# HDBSCAN's only real knob: the smallest group of prompts that counts as a
# cluster (anything sparser is noise). 8 is a sensible default for a few-hundred-
# to-few-thousand prompt corpus; overridable from the CLI (--min-cluster-size).
DEFAULT_MIN_CLUSTER_SIZE = 8

# How many distinctive terms / representative prompts to attach to a cluster.
_TOP_TERMS = 8
_TOP_REPRESENTATIVES = 3
_PREVIEW_CHARS = 100

# Diagnostic thresholds. Deliberately conservative: an audit that cries "merge!"
# on every incidental overlap is noise. Tuned to surface the clear signals.
_MERGE_CATEGORY_SHARE = 0.20  # a category must own >=20% of a cluster to "merge"
_SPLIT_TOP_SHARE = 0.50  # a category whose biggest cluster holds <50% is split...
_SPLIT_MIN_CLUSTERS = 3  # ...and that is spread over at least this many clusters
_SPLIT_MIN_PROMPTS = 12  # only audit splits for categories with enough mass
_CROSS_CUTTING_PURITY = 0.40  # below this dominant share, a cluster is "mixed"
_LATENT_OTHER_PURITY = 0.50  # an "other"-dominated cluster this pure is a theme

# Distinctive-term extraction works on word-ish tokens, lowercased, with the
# usual high-frequency FR + EN function words removed so a cluster's label is
# its *content*, not "the / and / pour / dans".
_TERM_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_STOPWORDS = frozenset(
    {
        # English
        "the", "and", "for", "with", "that", "this", "you", "your", "are", "but",
        "not", "can", "all", "any", "from", "have", "has", "was", "will", "what",
        "why", "how", "when", "which", "should", "would", "could", "into", "out",
        "then", "than", "them", "they", "there", "here", "its", "our", "use",
        "using", "make", "made", "get", "got", "want", "need", "like", "just",
        "also", "more", "most", "some", "one", "two", "now", "new", "add", "put",
        "let", "see", "via", "per", "yes", "okay",
        # French (accent-tolerant: the corpus is often unaccented)
        "les", "des", "une", "que", "qui", "pour", "dans", "avec", "sur", "pas",
        "est", "sont", "fait", "faire", "fais", "plus", "moins", "mais", "donc",
        "car", "comme", "tout", "tous", "toute", "cette", "ces", "son", "ses",
        "leur", "vous", "nous", "elle", "ici", "etre", "etait", "avoir", "deja",
        "encore", "bien", "tres", "peux", "peut", "veux", "veut", "oui", "non",
        "ton", "tes", "mon", "mes", "par", "aux",
    }
)


@dataclass(frozen=True)
class Cluster:
    """One natural cluster, labelled and cross-tabbed against the taxonomy."""

    cluster_id: int  # -1 is the HDBSCAN noise bucket
    size: int
    category_counts: dict[str, int]  # current category → count in this cluster
    dominant_category: str
    purity: float  # dominant_count / size (1.0 = a pure cluster)
    top_terms: list[str]
    representatives: list[str]  # prompt previews closest to the centroid


@dataclass(frozen=True)
class MergeSignal:
    """Categories the data lumps into a single cluster (taxonomy may over-split)."""

    cluster_id: int
    categories: list[str]
    top_terms: list[str]


@dataclass(frozen=True)
class SplitSignal:
    """One category the data scatters across many clusters (taxonomy may be coarse)."""

    category: str
    cluster_ids: list[int]
    top_share: float  # share of the category landing in its single biggest cluster


@dataclass(frozen=True)
class ThemeSignal:
    """A cluster worth a human's eye: cross-cutting or latent inside ``other``."""

    cluster_id: int
    kind: str  # "cross-cutting" | "latent-other"
    purity: float
    dominant_category: str
    top_terms: list[str]


@dataclass(frozen=True)
class AuditReport:
    """Everything the audit found, ready to render or serialize."""

    n_prompts: int
    n_clusters: int  # excluding noise
    noise: int
    categories: list[str]  # categories present, in display order (matrix columns)
    clusters: list[Cluster]  # non-noise, sorted by size desc
    merges: list[MergeSignal] = field(default_factory=list)
    splits: list[SplitSignal] = field(default_factory=list)
    themes: list[ThemeSignal] = field(default_factory=list)
    other_distribution: list[tuple[int, int]] = field(default_factory=list)  # (cluster_id, count)


# ── tokenization for distinctive terms ────────────────────────────────────────


def _content_tokens(text: str) -> list[str]:
    """Lowercased word-ish tokens (>=3 letters), function words removed."""
    return [t for t in _TERM_RE.findall(text.lower()) if t not in _STOPWORDS]


# ── clustering ────────────────────────────────────────────────────────────────


def _cluster_labels(vectors: FloatMatrix, min_cluster_size: int) -> npt.NDArray[np.intp]:
    """HDBSCAN cluster labels (``-1`` = noise) for L2-normalized ``vectors``.

    Euclidean distance on unit vectors is monotonic with cosine distance, so the
    default metric clusters by semantic similarity. HDBSCAN is deterministic
    given the data, which keeps the audit reproducible.
    """
    from sklearn.cluster import HDBSCAN

    n = vectors.shape[0]
    if n < max(2, min_cluster_size):
        return np.full(n, -1, dtype=np.intp)
    clusterer = HDBSCAN(min_cluster_size=max(2, min_cluster_size), metric="euclidean")
    labels: npt.NDArray[np.intp] = clusterer.fit_predict(np.asarray(vectors, dtype=np.float64))
    return labels


# ── cluster labelling ─────────────────────────────────────────────────────────


def _top_terms(
    labels: npt.NDArray[np.intp], tokens_per_doc: list[list[str]], cluster_ids: list[int]
) -> dict[int, list[str]]:
    """Most distinctive terms per cluster via class-based TF-IDF (BERTopic-style).

    A term scores high in a cluster when it is frequent *there* yet rare across
    the *other* clusters. Ties break alphabetically so the labels are stable.
    """
    # Per-cluster term frequencies and per-cluster document frequency of a term.
    tf: dict[int, Counter[str]] = {cid: Counter() for cid in cluster_ids}
    df: Counter[str] = Counter()  # in how many clusters a term appears at all
    for cid in cluster_ids:
        members = np.flatnonzero(labels == cid)
        seen: set[str] = set()
        for idx in members:
            counts = Counter(tokens_per_doc[idx])
            tf[cid].update(counts)
            seen.update(counts)
        for term in seen:
            df[term] += 1

    n_clusters = max(len(cluster_ids), 1)
    result: dict[int, list[str]] = {}
    for cid in cluster_ids:
        total = sum(tf[cid].values()) or 1
        scored = [
            (count / total * math.log(1.0 + n_clusters / df[term]), term)
            for term, count in tf[cid].items()
        ]
        scored.sort(key=lambda s: (-s[0], s[1]))
        result[cid] = [term for _score, term in scored[:_TOP_TERMS]]
    return result


def _representatives(
    labels: npt.NDArray[np.intp], vectors: FloatMatrix, previews: list[str], cluster_ids: list[int]
) -> dict[int, list[str]]:
    """The prompts closest to each cluster's centroid (its archetypes)."""
    result: dict[int, list[str]] = {}
    for cid in cluster_ids:
        members = np.flatnonzero(labels == cid)
        if members.size == 0:
            result[cid] = []
            continue
        centroid = vectors[members].mean(axis=0)
        # Members are unit vectors → dot with the centroid ranks by cosine.
        scores = vectors[members] @ centroid
        order = members[np.argsort(-scores, kind="stable")][:_TOP_REPRESENTATIVES]
        result[cid] = [_clip(previews[idx]) for idx in order]
    return result


def _clip(text: str) -> str:
    """A single-line, length-bounded preview for the report/CSV."""
    flat = " ".join(text.split())
    return flat[:_PREVIEW_CHARS] + ("…" if len(flat) > _PREVIEW_CHARS else "")


# ── diagnostics ───────────────────────────────────────────────────────────────


def _detect_merges(clusters: list[Cluster]) -> list[MergeSignal]:
    """Clusters that blend >=2 real categories (each owning a fair share)."""
    merges: list[MergeSignal] = []
    for cluster in clusters:
        if cluster.size == 0:
            continue
        blended = [
            cat
            for cat, count in cluster.category_counts.items()
            if cat != "other" and count / cluster.size >= _MERGE_CATEGORY_SHARE
        ]
        if len(blended) >= 2:
            blended.sort(key=lambda c: -cluster.category_counts[c])
            merges.append(
                MergeSignal(cluster.cluster_id, blended, cluster.top_terms)
            )
    return merges


def _detect_splits(
    clusters: list[Cluster], categories: list[str]
) -> list[SplitSignal]:
    """Categories scattered across many clusters with no dominant home."""
    # category → {cluster_id: count} across non-noise clusters.
    spread: dict[str, dict[int, int]] = {cat: {} for cat in categories}
    for cluster in clusters:
        for cat, count in cluster.category_counts.items():
            if count:
                spread.setdefault(cat, {})[cluster.cluster_id] = count

    splits: list[SplitSignal] = []
    for cat in categories:
        if cat in ("other", "followup", "notification"):
            continue  # catch-all / short-circuit buckets are not "split" signals
        per_cluster = spread.get(cat, {})
        clustered = sum(per_cluster.values())
        if clustered < _SPLIT_MIN_PROMPTS or len(per_cluster) < _SPLIT_MIN_CLUSTERS:
            continue
        top_share = max(per_cluster.values()) / clustered
        if top_share < _SPLIT_TOP_SHARE:
            ids = sorted(per_cluster, key=lambda c: -per_cluster[c])
            splits.append(SplitSignal(cat, ids, top_share))
    return splits


def _detect_themes(clusters: list[Cluster]) -> list[ThemeSignal]:
    """Mixed (cross-cutting) and ``other``-dominated (latent) clusters."""
    themes: list[ThemeSignal] = []
    for cluster in clusters:
        if cluster.dominant_category == "other" and cluster.purity >= _LATENT_OTHER_PURITY:
            themes.append(
                ThemeSignal(
                    cluster.cluster_id, "latent-other", cluster.purity,
                    cluster.dominant_category, cluster.top_terms,
                )
            )
        elif cluster.purity < _CROSS_CUTTING_PURITY:
            themes.append(
                ThemeSignal(
                    cluster.cluster_id, "cross-cutting", cluster.purity,
                    cluster.dominant_category, cluster.top_terms,
                )
            )
    return themes


# ── report assembly ───────────────────────────────────────────────────────────


def _build_clusters(
    labels: npt.NDArray[np.intp],
    cats: list[str],
    categories: list[str],
    terms: dict[int, list[str]],
    reps: dict[int, list[str]],
    cluster_ids: list[int],
) -> list[Cluster]:
    """Cross-tab each cluster against the taxonomy and pick its dominant label."""
    clusters: list[Cluster] = []
    for cid in cluster_ids:
        members = np.flatnonzero(labels == cid)
        counts: Counter[str] = Counter(cats[idx] for idx in members)
        size = int(members.size)
        # Dominant category: plurality, ties broken by taxonomy order (stable).
        dominant, dom_count = "", 0
        for cat in categories:
            if counts.get(cat, 0) > dom_count:
                dominant, dom_count = cat, counts[cat]
        clusters.append(
            Cluster(
                cluster_id=int(cid),
                size=size,
                category_counts={cat: counts.get(cat, 0) for cat in categories if counts.get(cat)},
                dominant_category=dominant or "(uncategorized)",
                purity=(dom_count / size) if size else 0.0,
                top_terms=terms.get(cid, []),
                representatives=reps.get(cid, []),
            )
        )
    return clusters


def build_report(
    embedder: Embedder,
    prompt_ids: list[str],
    texts: list[str],
    categories_by_id: dict[str, str],
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    cache_path: Path | None = None,
) -> AuditReport:
    """Embed, cluster and cross-tab the corpus into an :class:`AuditReport`.

    ``embedder`` is the injection seam (production: ``StaticEmbedder``; tests:
    ``HashingEmbedder``). When ``cache_path`` is given, the disk embedding cache
    is reused so the audit shares vectors with ``categorize --semantic``.
    """
    cleaned = [_NOISE_WRAPPER_RE.sub(" ", t) for t in texts]
    if cache_path is not None:
        from .embeddings import EmbeddingCache

        vectors = EmbeddingCache(cache_path, embedder).embed(prompt_ids, cleaned)
    else:
        vectors = embedder.embed(cleaned)

    labels = _cluster_labels(vectors, min_cluster_size)
    cats = [categories_by_id.get(pid, "") or "(uncategorized)" for pid in prompt_ids]
    tokens_per_doc = [_content_tokens(t) for t in cleaned]

    # Categories present, in canonical taxonomy order for stable matrix columns.
    from .categorize import _TIE_BREAK_ORDER

    present = {c for c in cats}
    ordered = [c for c in _TIE_BREAK_ORDER if c in present]
    categories = ordered + sorted(present - set(ordered) - {""})

    cluster_ids = sorted({int(c) for c in labels if c != -1})
    terms = _top_terms(labels, tokens_per_doc, cluster_ids)
    reps = _representatives(labels, vectors, texts, cluster_ids)
    clusters = _build_clusters(labels, cats, categories, terms, reps, cluster_ids)
    clusters.sort(key=lambda c: -c.size)

    noise = int(np.count_nonzero(labels == -1))
    merges = _detect_merges(clusters)
    splits = _detect_splits(clusters, categories)
    themes = _detect_themes(clusters)

    # How the current "other" bucket regroups across natural clusters.
    other_by_cluster: Counter[int] = Counter(
        int(labels[i]) for i, cat in enumerate(cats) if cat == "other" and labels[i] != -1
    )
    other_distribution = sorted(other_by_cluster.items(), key=lambda kv: -kv[1])

    return AuditReport(
        n_prompts=len(prompt_ids),
        n_clusters=len(cluster_ids),
        noise=noise,
        categories=categories,
        clusters=clusters,
        merges=merges,
        splits=splits,
        themes=themes,
        other_distribution=other_distribution,
    )


# ── rendering ─────────────────────────────────────────────────────────────────


def format_report(report: AuditReport) -> list[str]:
    """Human-readable report lines (printed and written to ``taxonomy_audit.txt``)."""
    lines: list[str] = []
    lines.append("Taxonomy audit — natural clusters vs the 13 categories")
    lines.append("=" * 60)
    pct = (report.noise / report.n_prompts * 100) if report.n_prompts else 0.0
    lines.append(
        f"{report.n_prompts} prompts → {report.n_clusters} clusters "
        f"+ {report.noise} noise ({pct:.0f}%)"
    )
    lines.append("")
    lines.append("DISCLAIMER: a pure diagnostic. Unsupervised clusters are unstable and")
    lines.append("hard to name; they inform a deliberate human revision, never perform one.")
    lines.append("This does NOT change categories.csv or the taxonomy.")
    lines.append("")

    lines.append("Clusters (size · dominant category · purity · terms)")
    lines.append("-" * 60)
    for c in report.clusters:
        lines.append(
            f"#{c.cluster_id:<3} n={c.size:<4} {c.dominant_category:<16} "
            f"purity={c.purity:.0%}  {', '.join(c.top_terms) or '(no terms)'}"
        )
        for rep in c.representatives:
            lines.append(f"        · {rep}")
    lines.append("")

    lines.append("Merge signals (categories the data lumps together)")
    lines.append("-" * 60)
    if report.merges:
        for m in report.merges:
            lines.append(
                f"  cluster #{m.cluster_id}: {', '.join(m.categories)}  "
                f"[{', '.join(m.top_terms[:5])}]"
            )
    else:
        lines.append("  none — no cluster blends two categories above threshold.")
    lines.append("")

    lines.append("Split signals (one category scattered across clusters)")
    lines.append("-" * 60)
    if report.splits:
        for s in report.splits:
            ids = ", ".join(f"#{i}" for i in s.cluster_ids)
            lines.append(
                f"  {s.category}: spread over {len(s.cluster_ids)} clusters "
                f"({ids}); biggest holds only {s.top_share:.0%}"
            )
    else:
        lines.append("  none — every category has a clear home cluster.")
    lines.append("")

    lines.append("Themes to eyeball (cross-cutting / latent in 'other')")
    lines.append("-" * 60)
    if report.themes:
        for t in report.themes:
            lines.append(
                f"  cluster #{t.cluster_id} [{t.kind}] purity={t.purity:.0%} "
                f"(dominant: {t.dominant_category})  {', '.join(t.top_terms[:5])}"
            )
    else:
        lines.append("  none.")
    lines.append("")

    lines.append("'other' bucket regrouped")
    lines.append("-" * 60)
    if report.other_distribution:
        for cid, count in report.other_distribution[:8]:
            lines.append(f"  cluster #{cid}: {count} prompts currently in 'other'")
    else:
        lines.append("  no 'other'-labelled prompts landed in a cluster.")
    return lines


def write_audit_csv(path: Path, report: AuditReport) -> None:
    """Write the alignment matrix + cluster labels (one row per cluster)."""
    columns = [
        "cluster_id", "size", "dominant_category", "purity",
        "top_terms", "representatives", *report.categories,
    ]
    rows: list[dict[str, Any]] = []
    for c in report.clusters:
        row: dict[str, Any] = {
            "cluster_id": c.cluster_id,
            "size": c.size,
            "dominant_category": c.dominant_category,
            "purity": round(c.purity, 4),
            "top_terms": escape_csv_formula(" ".join(c.top_terms)),
            "representatives": escape_csv_formula(" | ".join(c.representatives)),
        }
        for cat in report.categories:
            row[cat] = c.category_counts.get(cat, 0)
        rows.append(row)
    atomic_write_csv(path, columns, rows)


# ── entry point ───────────────────────────────────────────────────────────────


def run_audit(
    *,
    output_dir: str = "./output",
    embedder: Embedder | None = None,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> int:
    """Run the taxonomy audit; write the report + CSV, print to stdout.

    Returns ``-1`` when there is nothing to audit (no ``prompts.csv``, or no
    prompt carries any text), so the CLI can exit non-zero; ``0`` on success.
    ``embedder`` defaults to the real :class:`StaticEmbedder`; tests inject the
    deterministic ``HashingEmbedder``.
    """
    out = Path(output_dir)
    prompts_path = out / "prompts.csv"
    if not prompts_path.exists():
        print(
            f"No prompts file at {prompts_path}. Run `prompt-analytics extract` first.",
            file=sys.stderr,
        )
        return -1

    with prompts_path.open(encoding="utf-8", newline="") as fh:
        all_prompts = list(csv.DictReader(fh))
    texts_by_id = _load_texts(out / "prompts_text.csv")
    categories_csv = _load_categories(out / "categories.csv")
    categories_by_id = {pid: row.get("category", "") for pid, row in categories_csv.items()}

    prompt_ids: list[str] = []
    texts: list[str] = []
    for row in all_prompts:
        pid = row.get("prompt_id", "")
        if not pid or _is_pseudo(pid):
            continue
        text = (texts_by_id.get(pid) or row.get("prompt_preview") or "").strip()
        if not text:
            continue  # nothing to embed (extract --no-text) → skip, never cluster ""
        prompt_ids.append(pid)
        texts.append(text)

    if not prompt_ids:
        print(
            "No prompt text to audit. Re-run `prompt-analytics extract` without "
            "--no-text, then try again.",
            file=sys.stderr,
        )
        return -1

    if embedder is None:
        from .embeddings import StaticEmbedder

        embedder = StaticEmbedder()

    if not categories_by_id:
        print(
            "[note] no categories.csv yet — clustering still runs, but the "
            "alignment matrix and merge/split signals need `categorize` first.",
            file=sys.stderr,
        )

    report = build_report(
        embedder,
        prompt_ids,
        texts,
        categories_by_id,
        min_cluster_size=min_cluster_size,
        cache_path=out / "embeddings.npz",
    )

    lines = format_report(report)
    print("\n".join(lines))

    report_path = out / "taxonomy_audit.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = out / "taxonomy_audit.csv"
    write_audit_csv(csv_path, report)
    print(f"\nReport  → {report_path.resolve()}")
    print(f"Matrix  → {csv_path.resolve()}")
    return 0
