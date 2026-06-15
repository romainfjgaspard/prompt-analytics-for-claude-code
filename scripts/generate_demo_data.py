"""Generate a deterministic, synthetic demo dataset for the public dashboard.

The output (``demo_data/``) is committed and powers the live demo and the
README screenshots. **No real data is involved**: everything here is generated
from a fixed seed, so the dataset can be regenerated verbatim whenever the CSV
schema evolves (run ``python scripts/generate_demo_data.py``).

Shape (≈ the plan's 9.1 target, extended by plan v2 phase 1):

* ~6 months of history, ~80 sessions, ~800 prompts, 5 fictional projects;
* a realistic depth profile — session-opening prompts carry large fresh input
  and cache *writes* (expensive), later prompts ride cache *reads* (cheap), so
  the Session-depth meta-analysis tells the same story as on real data;
* **request grain** (V10): the generator works request by request and writes
  ``requests.csv``; ``tokens.csv`` is derived by summation, so the V7
  invariant (per-prompt sums identical) holds by construction;
* **subagents** (1.2): ~8% of prompts carry sidechain requests
  (``is_sidechain=1`` rows in both files), **TTL split** (1.3) and
  **post-compaction continuations** (1.4) are represented;
* every prompt categorized (heuristic-v2) with an observed 1-5 complexity;
* a ``quota_log`` with weekly seven-day cycles and short five-hour cycles.

The numbers are priced at read time by the analytics layer, exactly like real
data — this script writes raw token counts only (D3).
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prompt_analytics import schema

SEED = 20260611
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "demo_data"

PERIOD_START = datetime(2025, 12, 15, tzinfo=timezone.utc)
PERIOD_END = datetime(2026, 6, 10, tzinfo=timezone.utc)
N_SESSIONS = 80
TARGET_PROMPTS = 800

PROJECTS = [
    ("webapp-frontend", "main"),
    ("data-pipeline", "develop"),
    ("ml-experiments", "main"),
    ("infra-terraform", "main"),
    ("mobile-app", "feature/onboarding"),
]

# Four models, ordered cheapest-family-last is irrelevant here -- this list's
# order is what the weight vectors in `_pick_model` index into. Fable 5 is the
# newest premium model (priced ~2x Opus): used on relatively FEW prompts but, on
# real histories, a large slice of the bill -- the dataset must show that.
MODELS = ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]

# Category -> (weight, complexity range). Weights sum loosely to 1.
# Mix mirrors a real history classified by heuristic-v2 (phase 4): the agentic
# categories (review/test/docs/ops/followup) exist and "other" is a remainder,
# not the first bar.
CATEGORIES: dict[str, tuple[float, tuple[int, int]]] = {
    "debug": (0.16, (3, 5)),
    "implementation": (0.14, (2, 5)),
    "question": (0.12, (1, 3)),
    "ops": (0.10, (1, 3)),
    "other": (0.10, (1, 3)),
    "plan": (0.09, (2, 4)),
    "review": (0.08, (2, 5)),
    "test": (0.07, (2, 4)),
    "docs": (0.06, (1, 3)),
    "followup": (0.05, (1, 2)),
    "refactor": (0.03, (3, 5)),
}

PREVIEWS: dict[str, list[str]] = {
    "plan": [
        "Read PLAN.md and outline the steps to add the export pipeline",
        "Design the data model for the new billing module",
        "How should we structure the multi-tenant migration?",
    ],
    "implementation": [
        "Implement the CSV export endpoint with pagination",
        "Add the retry decorator to the API client",
        "Wire the new settings page into the router",
    ],
    "debug": [
        "The dashboard crashes with a tz-naive vs tz-aware error, fix it",
        "Why does the nightly job double-count rows?",
        "Tests fail on Windows only — investigate the path handling",
    ],
    "refactor": [
        "Refactor the parser into a streaming single pass",
        "Extract the duplicated cost logic into one helper",
        "Split the god-object service into focused classes",
    ],
    "review": [
        "Review the diff before I open the PR",
        "Audit the error handling in the export pipeline",
        "Vérifie que les totaux collent avec la facture",
    ],
    "test": [
        "Add unit tests for the quota parser edge cases",
        "Relance les tests d'intégration et regarde les régressions",
        "Raise the coverage of filters.py above 90%",
    ],
    "docs": [
        "Update the README and the changelog for the release",
        "Met à jour le fichier de status avant de fermer",
        "Write docstrings for the public analytics helpers",
    ],
    "ops": [
        "Commit the changes and push the branch",
        "Fais le merge de main et ouvre la PR",
        "Deploy to staging and run the smoke script",
    ],
    "question": [
        "What's the difference between cache_read and cache_write billing?",
        "Does uv lock pin transitive dependencies?",
        "Is there a way to stream large CSVs in pandas?",
    ],
    "followup": [
        "oui vas y",
        "ok go ahead",
        "continue with option 2",
    ],
    "other": [
        "Here are my notes from the standup, keep them in mind",
        "Same as yesterday but for the staging tenant",
        "anglais",
    ],
}

ENTRYPOINTS = ["cli", "cli", "vscode"]
MODES = ["default", "default", "plan"]
VERSIONS = ["2.1.85", "2.1.90", "2.2.0"]
STOP_REASONS = ["end_turn", "end_turn", "end_turn", "tool_use", "max_tokens"]


def _weighted_choice(rng: random.Random, weighted: dict[str, tuple[float, tuple[int, int]]]) -> str:
    names = list(weighted)
    weights = [weighted[n][0] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _pick_model(rng: random.Random, depth: int, category: str) -> str:
    """Premium models (Fable/Opus) for the expensive prompts, Haiku for cheap
    deep follow-ups. Weights index into ``MODELS`` = [fable, opus, sonnet, haiku].

    Fable rides the session-openers and the hard categories (plan/debug): a
    minority of prompts, but each is expensive *and* priced ~2x Opus, so its
    cost share ends up far above its prompt share -- the real-data signal.
    """
    if depth == 1:
        # Session openers: the costliest prompts (fresh input + big cache writes).
        return rng.choices(MODELS, weights=[0.30, 0.50, 0.13, 0.07], k=1)[0]
    if category in ("plan", "debug"):
        return rng.choices(MODELS, weights=[0.20, 0.50, 0.22, 0.08], k=1)[0]
    if depth >= 6:
        # Cheap deep follow-ups: mostly Haiku, little Sonnet, no Fable.
        return rng.choices(MODELS, weights=[0.02, 0.18, 0.25, 0.55], k=1)[0]
    return rng.choices(MODELS, weights=[0.08, 0.37, 0.30, 0.25], k=1)[0]


def _request_counts(
    rng: random.Random, depth: int, complexity: int, turn: int, n_turns: int
) -> dict[str, int]:
    """Raw token counts for ONE API request of a prompt.

    The first request carries the fresh input and the big cache writes
    (expensive); follow-up tool turns mostly re-read the cached context.
    """
    scale = 0.6 + 0.12 * complexity
    out_chunk = int(rng.randint(400, 1800) * scale)
    if turn == 1 and depth == 1:
        counts = {
            "input": rng.randint(6000, 16000),
            "output": out_chunk,
            "cache_read": rng.randint(0, 60000),
            "cache_write_5m": rng.randint(25000, 80000),
            "cache_write_1h": rng.randint(0, 45000),
        }
    elif turn == 1:
        ramp = min(depth, 30) / 30.0
        counts = {
            "input": rng.randint(150, 1400),
            "output": out_chunk,
            "cache_read": int(rng.randint(120000, 350000) * (0.5 + ramp)),
            "cache_write_5m": rng.randint(2000, 22000),
            "cache_write_1h": rng.randint(0, 8000),
        }
    else:
        # Tool turn: the context is hot, tool results land as small writes.
        ramp = min(depth, 30) / 30.0
        counts = {
            "input": rng.randint(0, 80),
            "output": out_chunk,
            "cache_read": int(rng.randint(120000, 380000) * (0.5 + ramp)),
            "cache_write_5m": rng.randint(500, 9000),
            "cache_write_1h": 0,
        }
    # Occasional web-search requests (billed per request, excluded from totals).
    if rng.random() < 0.10 / n_turns:
        counts["server_tool_use"] = rng.randint(1, 6)
    return {k: v for k, v in counts.items() if v > 0}


def _sidechain_request_counts(rng: random.Random) -> dict[str, int]:
    """Raw token counts for one subagent (sidechain) request."""
    counts = {
        "input": rng.randint(500, 3000),
        "output": rng.randint(300, 2000),
        "cache_read": rng.randint(0, 150000),
        "cache_write_5m": rng.randint(1000, 12000),
    }
    return {k: v for k, v in counts.items() if v > 0}


def _session_lengths(rng: random.Random) -> list[int]:
    """N prompt counts (one per session) summing to ~TARGET_PROMPTS."""
    lengths = [max(1, int(rng.lognormvariate(2.0, 0.7))) for _ in range(N_SESSIONS)]
    total = sum(lengths)
    # Rescale to hit the target without going below 1.
    factor = TARGET_PROMPTS / total
    lengths = [max(1, round(length * factor)) for length in lengths]
    return lengths


def _random_dt(rng: random.Random, start: datetime, end: datetime) -> datetime:
    span = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, span))


def _emit_requests(
    requests: list[dict[str, object]],
    tokens_acc: dict[tuple[str, str, str, str, int], int],
    *,
    session_id: str,
    prompt_id: str,
    base_time: datetime,
    reqs: list[tuple[str, int, int, dict[str, int]]],
) -> None:
    """Append request rows and accumulate their counts for tokens.csv.

    ``reqs`` is a list of ``(model, is_sidechain, post_compact, counts)``;
    tokens.csv is derived from these sums, so requests.csv and tokens.csv can
    never drift apart (the V7/V10 invariant, by construction).
    """
    cursor = base_time
    for index, (model, side, post_compact, counts) in enumerate(reqs, start=1):
        cursor += timedelta(seconds=20 + 17 * ((index * 7) % 5))
        requests.append(
            {
                "session_id": session_id,
                "prompt_id": prompt_id,
                "request_index": index,
                "timestamp": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "model": model,
                "stop_reason": "end_turn" if index == len(reqs) or side else "tool_use",
                "is_sidechain": side,
                "post_compact": post_compact,
                "input_tokens": counts.get("input", 0),
                "output_tokens": counts.get("output", 0),
                "cache_read_tokens": counts.get("cache_read", 0),
                "cache_write_5m_tokens": counts.get("cache_write_5m", 0),
                "cache_write_1h_tokens": counts.get("cache_write_1h", 0),
                "server_tool_use_requests": counts.get("server_tool_use", 0),
            }
        )
        for token_type, count in counts.items():
            if count:
                key = (session_id, prompt_id, model, token_type, side)
                tokens_acc[key] = tokens_acc.get(key, 0) + count


def generate() -> dict[str, list[dict[str, object]]]:
    """Build all rows in memory (deterministic for a fixed seed)."""
    rng = random.Random(SEED)
    lengths = _session_lengths(rng)

    sessions: list[dict[str, object]] = []
    prompts: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []
    categories: list[dict[str, object]] = []
    # (session_id, prompt_id, model, token_type, is_sidechain) -> count.
    tokens_acc: dict[tuple[str, str, str, str, int], int] = {}

    for s_index, length in enumerate(lengths):
        project, branch = PROJECTS[s_index % len(PROJECTS)]
        cwd = f"/home/dev/projects/{project}"
        session_id = f"demo-{s_index:03d}-{rng.randrange(16**8):08x}"
        session_start = _random_dt(rng, PERIOD_START, PERIOD_END)
        sessions.append(
            {
                "session_id": session_id,
                "start_date": session_start.strftime("%Y-%m-%d"),
                "project": project,
                "cwd": cwd,
                "git_branch": branch,
            }
        )

        cursor = session_start
        for depth in range(1, length + 1):
            cursor += timedelta(minutes=rng.randint(2, 40))
            category = _weighted_choice(rng, CATEGORIES)
            lo, hi = CATEGORIES[category][1]
            complexity = rng.randint(lo, hi)
            model = _pick_model(rng, depth, category)
            prompt_id = f"{session_id}:p{depth:03d}"
            preview = rng.choice(PREVIEWS[category])

            # Request grain (V10): one expensive opening turn + tool turns.
            n_turns = rng.randint(1, 3 + complexity)
            reqs: list[tuple[str, int, int, dict[str, int]]] = [
                (model, 0, 0, _request_counts(rng, depth, complexity, turn, n_turns))
                for turn in range(1, n_turns + 1)
            ]
            # ~8% of prompts delegate to a subagent (1.2): sidechain requests,
            # usually on a cheaper model, excluded from assistant_turns.
            if rng.random() < 0.08:
                # Subagents run on cheaper models -- a premium orchestrator
                # delegates the grunt work (Fable almost never a subagent).
                side_model = rng.choices(MODELS, weights=[0.03, 0.12, 0.30, 0.55], k=1)[0]
                for _ in range(rng.randint(1, 2)):
                    reqs.append((side_model, 1, 0, _sidechain_request_counts(rng)))

            prompts.append(
                {
                    "session_id": session_id,
                    "prompt_id": prompt_id,
                    "prompt_index": depth,
                    "timestamp": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "project": project,
                    "cwd": cwd,
                    "git_branch": branch,
                    "mode": rng.choice(MODES),
                    "entrypoint": rng.choice(ENTRYPOINTS),
                    "version": rng.choice(VERSIONS),
                    "model": model,
                    "char_count": len(preview) + rng.randint(20, 1200) * complexity,
                    "assistant_turns": n_turns,
                    "tool_calls": rng.randint(0, 4 * complexity),
                    "final_stop_reason": rng.choice(STOP_REASONS),
                    "prompt_preview": preview,
                }
            )
            categories.append(
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "complexity": complexity,
                    "classifier_model": "heuristic-v2",
                    "classified_at": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                }
            )
            _emit_requests(
                requests,
                tokens_acc,
                session_id=session_id,
                prompt_id=prompt_id,
                base_time=cursor,
                reqs=reqs,
            )

    # Session overhead (pseudo-prompts): a handful of resumed/compacted
    # sessions carry a ``:_continuation`` tail -- rows present in tokens.csv
    # and requests.csv only, with NO matching prompts.csv row, exactly like
    # real extracts. The dashboard must not lose them (N1). Every other one is
    # a post-compaction continuation (1.4). A dedicated RNG keeps the rest of
    # the dataset stable.
    overhead_rng = random.Random(SEED + 1)
    for tail_index, s_index in enumerate(range(0, N_SESSIONS, 16)):
        session_id = str(sessions[s_index]["session_id"])
        pseudo_id = schema.continuation_prompt_id(session_id)
        model = overhead_rng.choice(MODELS)
        post_compact = 1 if tail_index % 2 == 0 else 0
        if post_compact:
            # A compacted conversation carries a SMALL context (the summary,
            # ~25-60k tokens -- the real-data median is ~37k): the "after" of
            # a compaction event must be far below the "before", otherwise the
            # compaction analysis reads a negative reduction on demo data.
            # Rebuilding the cache from it is the expensive part.
            counts = {
                "cache_read": overhead_rng.randint(25_000, 60_000),
                "output": overhead_rng.randint(300, 2_000),
                "cache_write_5m": overhead_rng.randint(20_000, 90_000),
            }
        else:
            counts = {
                "cache_read": overhead_rng.randint(200_000, 900_000),
                "output": overhead_rng.randint(300, 2_000),
            }
        start_dates = str(sessions[s_index]["start_date"])
        base_time = datetime.strptime(start_dates, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ) + timedelta(hours=overhead_rng.randint(1, 20))
        _emit_requests(
            requests,
            tokens_acc,
            session_id=session_id,
            prompt_id=pseudo_id,
            base_time=base_time,
            reqs=[(model, 0, post_compact, counts)],
        )

    # tokens.csv derived from the same per-request counts (V7 by construction),
    # ordered like a real extract.
    token_order = {name: i for i, name in enumerate(schema.TOKEN_TYPES)}
    tokens: list[dict[str, object]] = [
        {
            "session_id": session_id,
            "prompt_id": prompt_id,
            "model": model,
            "token_type": token_type,
            "is_sidechain": side,
            "token_count": count,
        }
        for (session_id, prompt_id, model, token_type, side), count in sorted(
            tokens_acc.items(), key=lambda kv: (kv[0][1], kv[0][2], token_order[kv[0][3]], kv[0][4])
        )
    ]
    requests.sort(key=lambda r: (str(r["session_id"]), str(r["timestamp"]), str(r["prompt_id"])))

    token_types: list[dict[str, object]] = [
        {
            "token_type": tt,
            "label": schema.TOKEN_TYPE_LABELS[tt],
            "description": schema.TOKEN_TYPE_DESCRIPTIONS[tt],
        }
        for tt in schema.TOKEN_TYPES
    ]

    quota_log = _generate_quota(rng)

    return {
        "sessions": sessions,
        "prompts": prompts,
        "tokens": tokens,
        "requests": requests,
        "categories": categories,
        "token_types": token_types,
        "quota_log": quota_log,
    }


def _generate_quota(rng: random.Random) -> list[dict[str, object]]:
    """Daily snapshots: weekly seven-day cycle + short five-hour cycle."""
    rows: list[dict[str, object]] = []
    day = PERIOD_END - timedelta(days=45)
    while day <= PERIOD_END:
        # seven_day: ramps across the week, resets each Monday.
        dow = day.weekday()
        seven = min(100.0, 12 * dow + rng.uniform(0, 12))
        seven_reset = (day + timedelta(days=(7 - dow))).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # five_hour: sawtooth within the day.
        five = min(100.0, (day.hour % 5) * 18 + rng.uniform(5, 35))
        five_reset = day + timedelta(hours=(5 - day.hour % 5))
        sonnet = min(100.0, seven * 0.4 + rng.uniform(0, 10))
        snap = day.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        rows.append(
            {
                "snapshot_at": snap,
                "field": "five_hour",
                "utilization_pct": round(five, 1),
                "resets_at": five_reset.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            }
        )
        rows.append(
            {
                "snapshot_at": snap,
                "field": "seven_day",
                "utilization_pct": round(seven, 1),
                "resets_at": seven_reset.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            }
        )
        rows.append(
            {
                "snapshot_at": snap,
                "field": "seven_day_sonnet",
                "utilization_pct": round(sonnet, 1),
                "resets_at": seven_reset.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            }
        )
        day += timedelta(hours=rng.randint(10, 26))
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def main() -> None:
    """Generate and write the demo dataset to ``demo_data/``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = generate()
    _write_csv(OUT_DIR / "sessions.csv", schema.SESSIONS_COLS, data["sessions"])
    _write_csv(OUT_DIR / "prompts.csv", schema.PROMPTS_COLS, data["prompts"])
    _write_csv(OUT_DIR / "tokens.csv", schema.TOKENS_COLS, data["tokens"])
    _write_csv(OUT_DIR / "requests.csv", schema.REQUESTS_COLS, data["requests"])
    _write_csv(OUT_DIR / "token_types.csv", schema.TOKEN_TYPES_COLS, data["token_types"])
    _write_csv(OUT_DIR / "categories.csv", schema.CATEGORIES_COLS, data["categories"])
    _write_csv(OUT_DIR / "quota_log.csv", schema.QUOTA_LOG_COLS, data["quota_log"])

    # A config.yml so the dashboard's gated pages (categorization, quota) light up.
    (OUT_DIR / "config.yml").write_text(
        "features:\n  categorization: true\n  prompt_text: false\n  quota_snapshot: true\n",
        encoding="utf-8",
    )

    print(
        f"Wrote demo dataset to {OUT_DIR}: "
        f"{len(data['sessions'])} sessions, {len(data['prompts'])} prompts, "
        f"{len(data['tokens'])} token rows, {len(data['requests'])} request rows, "
        f"{len(data['quota_log'])} quota snapshots."
    )


if __name__ == "__main__":
    main()
