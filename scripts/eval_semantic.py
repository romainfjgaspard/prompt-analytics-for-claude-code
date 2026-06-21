"""Evaluate & calibrate the offline semantic classifier (Axe B1.3).

This is a **development-time** tool (run by us, once, to make a decision), *not*
something an end user ever touches: they run ``categorize`` and get categories
automatically, with zero manual labelling. The script measures whether the
semantic classifier is good enough to become the default, and calibrates its two
knobs (``τ`` and the lexical ``prime_weight``).

Two yardsticks, per the plan (§5 / §8):

1. **Litmus set** — ``scripts/eval_litmus.yml``, a small set of hand-curated hard
   cases (gold labels written by us), with the real architect complaint
   (test↔implementation) front and centre. Deterministic, offline, the anchor.
2. **LLM judge** — a "silver" reference: the existing ``--llm`` mode (here Azure
   OpenAI, the only reachable LLM) labels a larger sample of the demo corpus, and
   we measure which offline classifier (heuristic vs semantic) agrees with it
   more. Needs an API key; skipped cleanly when absent.

Outputs, to stdout and an optional ``--out`` file: the heuristic-vs-semantic
agreement matrix and global agreement, litmus pass/fail for both, judge
agreement for both, a sample of divergences, a τ/prime_weight calibration grid,
and a promotion recommendation.

Usage::

    uv run python scripts/eval_semantic.py                 # demo_data, no judge
    uv run python scripts/eval_semantic.py --judge --sample 150
    uv run python scripts/eval_semantic.py --judge --provider azure --out eval.txt

The pure functions (``load_litmus``, ``agreement_matrix``, ``calibrate``, …) are
import-safe and exercised by ``tests/test_eval_semantic.py`` with the
deterministic ``HashingEmbedder`` — no network in the test suite.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prompt_analytics.categorize import (  # noqa: E402
    DEFAULT_PRIME_WEIGHT,
    DEFAULT_TAU,
    DEFAULT_TOP_K,
    SemanticClassifier,
    _classify_heuristic,
    _is_pseudo,
    _load_texts,
    _Prepared,
    build_client,
)
from prompt_analytics.embeddings import Embedder, FloatMatrix  # noqa: E402

LITMUS_PATH = REPO_ROOT / "scripts" / "eval_litmus.yml"
DEMO_DIR = REPO_ROOT / "demo_data"

# Calibration grids: τ drives the "other" volume, prime_weight the pull of the
# lexical (ops/feedback) prime against the semantic categories.
TAU_GRID = [round(0.20 + 0.025 * i, 3) for i in range(11)]  # 0.200 … 0.450
PRIME_GRID = [round(0.40 + 0.10 * i, 2) for i in range(7)]  # 0.40 … 1.00


# ── data structures ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LitmusCase:
    text: str
    gold: str
    note: str = ""


@dataclass
class LitmusResult:
    accuracy: float
    rows: list[tuple[str, str, str, bool]]  # (text, gold, predicted, ok)


@dataclass
class AgreementResult:
    matrix: dict[tuple[str, str], int]  # (heuristic_label, semantic_label) -> count
    n_total: int
    n_agree: int

    @property
    def agreement(self) -> float:
        return self.n_agree / self.n_total if self.n_total else 0.0


@dataclass
class CalibrationResult:
    best_tau: float
    best_prime_weight: float
    best_litmus_acc: float
    best_judge_agree: float | None
    grid: list[tuple[float, float, float, float | None]]  # tau, pw, litmus, judge
    used_judge: bool


@dataclass
class JudgeResult:
    labels: dict[str, str]  # pid -> judge category
    heuristic_acc: float
    semantic_acc: float
    n: int


@dataclass
class EvalReport:
    n_prompts: int
    litmus_heuristic: LitmusResult
    litmus_semantic: LitmusResult
    agreement: AgreementResult
    calibration: CalibrationResult
    judge: JudgeResult | None
    reclassified_other: list[tuple[str, str]]  # (text, semantic_label)
    contradictions: list[tuple[str, str, str]]  # (text, heuristic, semantic)
    params: tuple[float, float, int]  # tau, prime_weight, top_k
    extras: dict[str, str] = field(default_factory=dict)


# ── loading ──────────────────────────────────────────────────────────────────


def load_litmus(path: str | Path = LITMUS_PATH) -> list[LitmusCase]:
    """Load the committed litmus fixture (hand-curated hard cases)."""
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: missing or empty 'cases' list")
    out: list[LitmusCase] = []
    for c in cases:
        out.append(
            LitmusCase(text=str(c["text"]), gold=str(c["gold"]), note=str(c.get("note", "")))
        )
    return out


def load_demo_texts(demo_dir: str | Path = DEMO_DIR) -> dict[str, str]:
    """Return ``prompt_id -> text`` for the real (non-pseudo) demo prompts.

    Mirrors how ``categorize`` reads text: the stored ``prompts_text.csv`` when
    present, otherwise the ``prompt_preview`` carried in ``prompts.csv``.
    """
    demo = Path(demo_dir)
    texts = _load_texts(demo / "prompts_text.csv")
    out: dict[str, str] = {}
    with (demo / "prompts.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = row.get("prompt_id", "")
            if not pid or _is_pseudo(pid):
                continue
            text = (texts.get(pid) or row.get("prompt_preview") or "").strip()
            if text:
                out[pid] = text
    return out


# ── classification helpers ─────────────────────────────────────────────────────


def heuristic_labels(texts: dict[str, str]) -> dict[str, str]:
    return {pid: _classify_heuristic(text) for pid, text in texts.items()}


def prepare_and_embed(
    clf: SemanticClassifier, texts: list[str]
) -> tuple[list[_Prepared], FloatMatrix]:
    """Prepare (strip chrome) and embed a batch once, for reuse across the grid.

    Embedding is the expensive step; τ and prime_weight only affect *scoring*, so
    we embed once and re-label cheaply by mutating the classifier's two knobs.
    """
    preps = [clf.prepare(t) for t in texts]
    vectors = clf.embedder.embed([p.text for p in preps])
    return preps, vectors


def label_all(clf: SemanticClassifier, preps: list[_Prepared], vectors: FloatMatrix) -> list[str]:
    out: list[str] = []
    for i, prep in enumerate(preps):
        vec = vectors[i] if vectors.shape[0] > i else np.zeros((0,), dtype=np.float32)
        out.append(clf.label(prep, vec))
    return out


def semantic_labels(clf: SemanticClassifier, texts: dict[str, str]) -> dict[str, str]:
    ids = list(texts)
    preps, vectors = prepare_and_embed(clf, [texts[p] for p in ids])
    labels = label_all(clf, preps, vectors)
    return dict(zip(ids, labels, strict=True))


# ── agreement & divergences ─────────────────────────────────────────────────────


def agreement_matrix(a: dict[str, str], b: dict[str, str]) -> AgreementResult:
    """Confusion of two label maps over their shared prompt ids."""
    shared = [pid for pid in a if pid in b]
    matrix: dict[tuple[str, str], int] = {}
    n_agree = 0
    for pid in shared:
        key = (a[pid], b[pid])
        matrix[key] = matrix.get(key, 0) + 1
        if a[pid] == b[pid]:
            n_agree += 1
    return AgreementResult(matrix=matrix, n_total=len(shared), n_agree=n_agree)


def sample_divergences(
    texts: dict[str, str],
    heuristic: dict[str, str],
    semantic: dict[str, str],
    *,
    limit: int = 12,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Two illustrative buckets, sorted by prompt id for determinism.

    * ``reclassified_other`` — heuristic said ``other``, semantic pulled it into a
      real category (the "clean up other" effect).
    * ``contradictions`` — both committed to a category, but a *different* one.
    """
    reclassified: list[tuple[str, str]] = []
    contradictions: list[tuple[str, str, str]] = []
    seen_re: set[str] = set()
    seen_co: set[str] = set()
    for pid in sorted(texts):
        h, s = heuristic.get(pid), semantic.get(pid)
        if h is None or s is None or h == s:
            continue
        text = texts[pid]
        if h == "other" and s != "other" and text not in seen_re:
            seen_re.add(text)
            reclassified.append((text, s))
        elif h != "other" and s != "other" and text not in seen_co:
            seen_co.add(text)
            contradictions.append((text, h, s))
    return reclassified[:limit], contradictions[:limit]


# ── litmus & calibration ─────────────────────────────────────────────────────────


def evaluate_litmus(cases: list[LitmusCase], labels: list[str]) -> LitmusResult:
    rows = [(c.text, c.gold, pred, c.gold == pred) for c, pred in zip(cases, labels, strict=True)]
    acc = sum(1 for *_rest, ok in rows if ok) / len(rows) if rows else 0.0
    return LitmusResult(accuracy=acc, rows=rows)


def calibrate(
    clf: SemanticClassifier,
    litmus_preps: list[_Prepared],
    litmus_vectors: FloatMatrix,
    litmus_golds: list[str],
    *,
    judge_preps: list[_Prepared] | None = None,
    judge_vectors: FloatMatrix | None = None,
    judge_golds: list[str] | None = None,
    tau_grid: list[float] = TAU_GRID,
    prime_grid: list[float] = PRIME_GRID,
) -> CalibrationResult:
    """Grid-search τ × prime_weight, maximising agreement.

    Primary objective = litmus accuracy (deterministic, the anchor); the LLM
    judge agreement on the demo sample is the tie-breaker when available. Ties on
    both are broken toward a *higher* τ (more conservative "other") then a lower
    prime_weight (let the semantic signal lead), so the choice is reproducible.
    """
    use_judge = judge_preps is not None and judge_vectors is not None and judge_golds is not None
    grid: list[tuple[float, float, float, float | None]] = []
    best: tuple[float, float, float, float | None] | None = None
    best_key: tuple[float, float, float, float] | None = None

    saved = (clf.tau, clf.prime_weight)
    for pw in prime_grid:
        for tau in tau_grid:
            clf.tau, clf.prime_weight = tau, pw
            lit_pred = label_all(clf, litmus_preps, litmus_vectors)
            lit_hits = sum(1 for p, g in zip(lit_pred, litmus_golds, strict=True) if p == g)
            lit_acc = lit_hits / (len(litmus_golds) or 1)
            judge_agree: float | None = None
            if use_judge:
                assert judge_preps is not None and judge_vectors is not None
                assert judge_golds is not None
                jp = label_all(clf, judge_preps, judge_vectors)
                judge_hits = sum(1 for p, g in zip(jp, judge_golds, strict=True) if p == g)
                judge_agree = judge_hits / (len(judge_golds) or 1)
            grid.append((tau, pw, lit_acc, judge_agree))
            # Lexicographic objective; tie-breaks make the pick deterministic.
            key = (lit_acc, judge_agree if judge_agree is not None else 0.0, tau, -pw)
            if best_key is None or key > best_key:
                best_key, best = key, (tau, pw, lit_acc, judge_agree)
    clf.tau, clf.prime_weight = saved
    assert best is not None
    return CalibrationResult(
        best_tau=best[0],
        best_prime_weight=best[1],
        best_litmus_acc=best[2],
        best_judge_agree=best[3],
        grid=grid,
        used_judge=use_judge,
    )


# ── LLM judge ────────────────────────────────────────────────────────────────


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324  (cache key, not security)


def run_judge(
    client: object,
    items: list[tuple[str, str]],
    *,
    cache_path: Path | None = None,
) -> dict[str, str]:
    """Label ``items`` (pid, text) with the LLM judge, caching by text hash.

    The cache (a JSON map ``text-hash -> category``) lets re-runs and the
    calibration reuse judge labels without re-spending tokens. It lives under the
    git-ignored ``output/`` by default and is never committed.
    """
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}

    out: dict[str, str] = {}
    dirty = False
    for n, (pid, text) in enumerate(items, start=1):
        key = _text_key(text)
        cat = cache.get(key)
        if cat is None:
            cat, _comp = client.classify(text)  # type: ignore[attr-defined]
            if not cat:
                continue  # transient failure exhausted retries → skip this one
            cache[key] = cat
            dirty = True
        out[pid] = cat
        if n % 25 == 0 or n == len(items):
            print(f"  judge {n}/{len(items)}", file=sys.stderr)
    if dirty and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return out


def judge_accuracies(
    judge: dict[str, str], heuristic: dict[str, str], semantic: dict[str, str]
) -> JudgeResult:
    shared = [pid for pid in judge if pid in heuristic and pid in semantic]
    if not shared:
        return JudgeResult(labels=judge, heuristic_acc=0.0, semantic_acc=0.0, n=0)
    h = sum(1 for pid in shared if heuristic[pid] == judge[pid]) / len(shared)
    s = sum(1 for pid in shared if semantic[pid] == judge[pid]) / len(shared)
    return JudgeResult(labels=judge, heuristic_acc=h, semantic_acc=s, n=len(shared))


# ── reporting ────────────────────────────────────────────────────────────────


def _bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def render_report(report: EvalReport) -> str:  # noqa: C901 - linear, readable top-down
    tau, pw, top_k = report.params
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("SEMANTIC CLASSIFIER EVALUATION (Axe B1.3) — dev-time, offline runtime")
    add("=" * 78)
    add(
        f"Corpus: {report.n_prompts} demo prompts   |   current defaults: "
        f"τ={tau}  prime_weight={pw}  top_k={top_k}"
    )
    for k, v in report.extras.items():
        add(f"  {k}: {v}")
    add("")

    # 1. Litmus
    add("── 1. LITMUS SET (hand-curated hard cases, gold = our judgment) ──────────")
    add(
        f"  heuristic: {report.litmus_heuristic.accuracy:6.1%}   "
        f"semantic: {report.litmus_semantic.accuracy:6.1%}   "
        f"({len(report.litmus_semantic.rows)} cases)"
    )
    add("")
    add(f"  {'gold':>15}  {'heuristic':>14}  {'semantic':>14}   prompt")
    for (text, gold, h_pred, _), (_, _, s_pred, _) in zip(
        report.litmus_heuristic.rows, report.litmus_semantic.rows, strict=True
    ):
        hmk = "✓" if h_pred == gold else "✗"
        smk = "✓" if s_pred == gold else "✗"
        snippet = text if len(text) <= 46 else text[:43] + "…"
        add(f"  {gold:>15}  {hmk} {h_pred:>12}  {smk} {s_pred:>12}   {snippet}")
    add("")

    # 2. Heuristic vs semantic agreement
    ag = report.agreement
    add("── 2. HEURISTIC vs SEMANTIC (whole demo corpus) ─────────────────────────")
    add(
        f"  global agreement: {ag.agreement:6.1%}  ({ag.n_agree}/{ag.n_total})  {_bar(ag.agreement)}"
    )
    add("")
    # per-heuristic-category breakdown: where does semantic send each bucket?
    by_h: dict[str, dict[str, int]] = {}
    for (h, s), c in ag.matrix.items():
        by_h.setdefault(h, {})[s] = c
    add("  where each heuristic bucket lands under semantic (top shifts):")
    for h in sorted(by_h, key=lambda k: -sum(by_h[k].values())):
        dests = by_h[h]
        total = sum(dests.values())
        same = dests.get(h, 0)
        moved = total - same
        if moved == 0:
            continue
        top = sorted((d for d in dests.items() if d[0] != h), key=lambda kv: -kv[1])[:3]
        movers = ", ".join(f"{d}:{c}" for d, c in top)
        add(f"    {h:>15} ({total:>3})  kept {same:>3}  moved {moved:>3}  → {movers}")
    add("")

    # 3. divergence samples
    add("── 3. DIVERGENCE SAMPLES ────────────────────────────────────────────────")
    add(f"  (a) semantic rescued from heuristic 'other'  ({len(report.reclassified_other)} shown):")
    for text, s in report.reclassified_other:
        snippet = text if len(text) <= 56 else text[:53] + "…"
        add(f"      → {s:>15}   {snippet}")
    add(f"  (b) both committed but disagree  ({len(report.contradictions)} shown):")
    for text, h, s in report.contradictions:
        snippet = text if len(text) <= 46 else text[:43] + "…"
        add(f"      heur={h:>14}  sem={s:>14}   {snippet}")
    add("")

    # 4. LLM judge
    add("── 4. LLM JUDGE (silver reference, dev-time) ────────────────────────────")
    if report.judge is None:
        add("  (skipped — no LLM key, or --judge not set)")
    else:
        j = report.judge
        add(f"  judged {j.n} prompts.  agreement with judge:")
        add(f"    heuristic: {j.heuristic_acc:6.1%}  {_bar(j.heuristic_acc)}")
        add(f"    semantic : {j.semantic_acc:6.1%}  {_bar(j.semantic_acc)}")
    add("")

    # 5. calibration
    cal = report.calibration
    add("── 5. CALIBRATION (grid search τ × prime_weight) ────────────────────────")
    obj = "litmus acc, tie-break = judge agreement" if cal.used_judge else "litmus accuracy"
    add(f"  objective: maximise {obj}")
    add(
        f"  BEST: τ={cal.best_tau}  prime_weight={cal.best_prime_weight}  "
        f"→ litmus {cal.best_litmus_acc:.1%}"
        + (f"  judge {cal.best_judge_agree:.1%}" if cal.best_judge_agree is not None else "")
    )
    add("")
    # compact grid: best τ per prime_weight row
    add("  best τ per prime_weight (litmus acc):")
    by_pw: dict[float, list[tuple[float, float, float | None]]] = {}
    for t, p, lit, jg in cal.grid:
        by_pw.setdefault(p, []).append((t, lit, jg))
    for p in sorted(by_pw):
        row = by_pw[p]
        best_t, best_lit, _ = max(row, key=lambda r: (r[1], r[2] or 0.0, r[0]))
        add(f"    prime_weight={p:.2f}:  best τ={best_t:.3f}  litmus={best_lit:.1%}")
    add("")

    # 6. verdict
    add("── 6. PROMOTION VERDICT ─────────────────────────────────────────────────")
    lit_ok = report.litmus_semantic.accuracy >= report.litmus_heuristic.accuracy
    judge_ok = (
        report.judge is None or report.judge.semantic_acc + 1e-9 >= report.judge.heuristic_acc
    )
    if report.judge is None:
        add("  judge: not run → decision rests on the litmus + manual review.")
    recommend = lit_ok and judge_ok
    add(
        f"  semantic ≥ heuristic on litmus : {'YES' if lit_ok else 'NO'} "
        f"({report.litmus_semantic.accuracy:.1%} vs {report.litmus_heuristic.accuracy:.1%})"
    )
    if report.judge is not None:
        add(
            f"  semantic ≥ heuristic on judge  : {'YES' if judge_ok else 'NO'} "
            f"({report.judge.semantic_acc:.1%} vs {report.judge.heuristic_acc:.1%})"
        )
    add("")
    add(
        f"  → RECOMMENDATION: {'PROMOTE semantic to default' if recommend else 'KEEP heuristic default'}"
    )
    add("    (final call is the developer's, recorded in the PSP status doc.)")
    add("=" * 78)
    return "\n".join(lines)


# ── orchestration ──────────────────────────────────────────────────────────────


def build_report(
    *,
    embedder: Embedder,
    texts: dict[str, str],
    litmus: list[LitmusCase],
    tau: float = DEFAULT_TAU,
    prime_weight: float = DEFAULT_PRIME_WEIGHT,
    top_k: int = DEFAULT_TOP_K,
    judge_client: object | None = None,
    judge_sample: int = 150,
    judge_cache: Path | None = None,
    extras: dict[str, str] | None = None,
) -> EvalReport:
    """Assemble the full evaluation (pure given an embedder + optional judge)."""
    clf = SemanticClassifier(embedder, tau=tau, prime_weight=prime_weight, top_k=top_k)

    heur = heuristic_labels(texts)
    sem = semantic_labels(clf, texts)
    agreement = agreement_matrix(heur, sem)
    reclassified, contradictions = sample_divergences(texts, heur, sem)

    # Litmus, embedded once and reused by the calibration grid.
    lit_texts = [c.text for c in litmus]
    lit_golds = [c.gold for c in litmus]
    lit_preps, lit_vectors = prepare_and_embed(clf, lit_texts)
    lit_h = evaluate_litmus(litmus, [_classify_heuristic(t) for t in lit_texts])
    lit_s = evaluate_litmus(litmus, label_all(clf, lit_preps, lit_vectors))

    # LLM judge on a deterministic sample (sorted ids) of the demo corpus.
    judge: JudgeResult | None = None
    judge_preps = judge_vectors = None
    judge_golds: list[str] | None = None
    if judge_client is not None:
        sample_ids = sorted(texts)[:judge_sample]
        items = [(pid, texts[pid]) for pid in sample_ids]
        judge_labels = run_judge(judge_client, items, cache_path=judge_cache)
        judge = judge_accuracies(judge_labels, heur, sem)
        jids = [pid for pid in sample_ids if pid in judge_labels]
        if jids:
            judge_preps, judge_vectors = prepare_and_embed(clf, [texts[p] for p in jids])
            judge_golds = [judge_labels[p] for p in jids]

    calibration = calibrate(
        clf,
        lit_preps,
        lit_vectors,
        lit_golds,
        judge_preps=judge_preps,
        judge_vectors=judge_vectors,
        judge_golds=judge_golds,
    )

    return EvalReport(
        n_prompts=len(texts),
        litmus_heuristic=lit_h,
        litmus_semantic=lit_s,
        agreement=agreement,
        calibration=calibration,
        judge=judge,
        reclassified_other=reclassified,
        contradictions=contradictions,
        params=(tau, prime_weight, top_k),
        extras=extras or {},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate/calibrate the semantic classifier (B1.3)."
    )
    parser.add_argument(
        "--demo-dir", default=str(DEMO_DIR), help="Corpus dir (default: demo_data)."
    )
    parser.add_argument(
        "--judge", action="store_true", help="Run the LLM judge (needs an API key)."
    )
    parser.add_argument(
        "--provider",
        default="azure",
        choices=["auto", "anthropic", "openrouter", "ollama", "azure"],
        help="LLM provider for the judge (default: azure).",
    )
    parser.add_argument("--sample", type=int, default=150, help="Judge sample size (default: 150).")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="τ to evaluate at.")
    parser.add_argument(
        "--prime-weight",
        type=float,
        default=DEFAULT_PRIME_WEIGHT,
        help="prime_weight to evaluate at.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Prototype top-k.")
    parser.add_argument(
        "--hashing", action="store_true", help="Use the HashingEmbedder (smoke/offline, no model)."
    )
    parser.add_argument("--out", default="", help="Also write the report to this file.")
    parser.add_argument(
        "--judge-cache",
        default=str(REPO_ROOT / "output" / "eval_judge_cache.json"),
        help="JSON cache of judge labels (git-ignored; never committed).",
    )
    args = parser.parse_args(argv)

    # The report uses τ, box-drawing and check marks; the Windows console defaults
    # to cp1252 and would choke on them. Force UTF-8 on the streams we print to.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    texts = load_demo_texts(args.demo_dir)
    if not texts:
        print(f"No prompts found under {args.demo_dir}.", file=sys.stderr)
        return 1
    litmus = load_litmus()

    if args.hashing:
        from prompt_analytics.embeddings import HashingEmbedder

        embedder: Embedder = HashingEmbedder()
        extras = {"embedder": "HashingEmbedder (smoke — numbers not meaningful)"}
    else:
        from prompt_analytics.embeddings import StaticEmbedder

        embedder = StaticEmbedder()
        extras = {"embedder": embedder.name}

    judge_client = None
    if args.judge:
        judge_client = build_client(provider=args.provider)
        if judge_client is None:
            print(
                "[warn] no judge client available; running without the LLM judge.", file=sys.stderr
            )
        else:
            extras["judge"] = (
                f"{type(judge_client).__name__} ({getattr(judge_client, 'model', '?')})"
            )

    report = build_report(
        embedder=embedder,
        texts=texts,
        litmus=litmus,
        tau=args.tau,
        prime_weight=args.prime_weight,
        top_k=args.top_k,
        judge_client=judge_client,
        judge_sample=args.sample,
        judge_cache=Path(args.judge_cache) if args.judge else None,
        extras=extras,
    )
    text = render_report(report)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n[written] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
