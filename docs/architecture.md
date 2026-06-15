# Architecture

A small, layered, quasi-stdlib pipeline. Raw JSONL logs are parsed into typed
records; costs are computed at read time from those records; the same analytics
layer feeds the CLI, the dashboard, and the CSV exports.

## Data flow

```
~/.claude/projects/**/*.jsonl     (CLAUDE_CONFIG_DIR honored, paths.py)
        │
        │  extract.collect()  — streaming parse, one pass per file
        │  • per-file parse cache (%LOCALAPPDATA% / $XDG_CACHE_HOME,
        │    key=path+mtime+size, orphan entries GC'd each run)
        │  • global dedup on message.id+requestId (largest snapshot wins)
        │  • promptId / uuid→parentUuid attribution
        │  • filtering (isMeta, <command-*>, interruptions, compaction)
        ▼
   schema.py shapes  (ParsedFile → SessionRow / PromptRow / TokenRow /
        │             RequestRow, raw counts)
        ├───────────────► extract.run_extract()
        │                 atomic_write_csv → 6 normalized CSVs in ./output
        │                 + extract_meta.json (the --since/--until window)
        ▼
   analytics.Dataset  (sessions × prompts × tokens × requests × categories)
        │  analytics.CostEngine(provider): raw counts + pricing.yml → USD
        │  (loud about unpriced models; never silently zeroed)
        │
        ├──────────────┬───────────────────┬─────────────────────┐
        ▼              ▼                   ▼                     ▼
     cli.py        dashboard/          export --flat         compare /
   (rich tables,   (pandas reshape,    (flat.csv,            by-project /
    json, csv)      ECharts, Streamlit) one row/prompt)       sessions --depth …
```

The key decision (PLAN.md D2/D3): **`extract` is an export, not a prerequisite.**
Every analysis command calls `analytics.load_dataset(output_dir)`, which uses the
output CSVs when they are fresh (CSV mtimes newer than the JSONL, and the
`tokens.csv`/`requests.csv` headers are the current schema) and otherwise parses
the JSONL live in memory — fast thanks to the per-file parse cache. So the numbers are always
current, and changing `pricing.yml` never requires a re-extract because costs are
computed from raw counts at read time.

## Modules

| Module | Responsibility |
| --- | --- |
| `schema.py` | The data contract: token-type keys, every CSV's columns, and the `TypedDict` shapes exchanged between parser and aggregation. **Single source of truth.** |
| `extract.py` | `collect()` parses JSONL → in-memory typed records; `run_extract()` adds atomic CSV writes and the extraction report. Owns the per-file parse cache. |
| `pricing.py` | Loads/validates `data/pricing.yml`; `get_model_pricing()` does exact→longest-prefix lookup (suffixes stripped); `get_per_request()` (server-tool requests), `load_plans()` (flat-rate subscriptions), `is_long_context()` (base-rate flag); `load_pricing`/`clear_cache`. The grid has three sections: `providers` (per-token + optional `per_request`), `plans` (`label`/`monthly_usd`), `updated_at`. |
| `analytics.py` | `Dataset`, `load_dataset`/`dataset_from_csvs`, `CostEngine`, and every aggregation (`summary`, `by_project`, `session_depth`, the request-grain analyses `context_growth`/`ttl_losses`/`compactions`/`session_overhead`, `model_category`/`recommendations`/`burn_rate`, `break_even`, `compare_providers`, `flat_export`, …). No streamlit/pandas. `CostEngine` prices per-token types from the grid and `server_tool_use` from `per_request`, and tracks unpriced + long-context models for loud notes. |
| `categorize.py` | Weighted FR+EN regex classifier (`heuristic-v2`, 11 categories) + observed complexity 1–5; opt-in LLM clients (Anthropic / OpenRouter / Ollama). Writes `categories.csv`. |
| `snapshot.py` | Reads the OAuth quota endpoint; appends to `quota_log.csv`. |
| `paths.py` | The two machine-dependent roots, shared by extract/cli/snapshot: the Claude dir (`~/.claude`, `CLAUDE_CONFIG_DIR` honored) and the per-user cache dir (`%LOCALAPPDATA%` / `$XDG_CACHE_HOME`). |
| `storage.py` | `atomic_write_csv` / `atomic_write_json` (tmp + `os.replace`), formula-injection escaping, append-with-header-if-empty. |
| `config.py` | Pure read of `config.yml` (+ `config init`); no side-effecting file creation. |
| `render.py` | Turns an analytics `TableResult` into a rich table / CSV / JSON. The only place display formatting (`$`, `%`, separators) lives. |
| `cli.py` / `__main__.py` | Argument parsing, dispatch, exit codes. |
| `dashboard/` | Streamlit app: `data.py` builds a `Dataset` via `analytics.dataset_from_csvs` and reshapes to pandas frames; `filters.py` (cross-filters); `echarts.py` (the Apache ECharts render layer + click/brush cross-filtering); `theme.py` (color semantics + typography); one script per page under `pages/`. An optional extra. |

## CSV contract

The canonical output is six normalized CSVs (plus `quota_log.csv` and the
`extract_meta.json` marker), keyed by `prompt_id` / `session_id`. **The
authoritative column layout is
[`prompt_analytics/schema.py`](../prompt_analytics/schema.py)** — `SESSIONS_COLS`,
`PROMPTS_COLS`, `TOKENS_COLS`, `REQUESTS_COLS`, `TOKEN_TYPES_COLS`,
`PROMPT_TEXT_COLS`, `CATEGORIES_COLS`, `QUOTA_LOG_COLS`. Every writer and reader
imports from there, so adding a column is a one-line change that propagates.

| File | Key | Grain |
| --- | --- | --- |
| `sessions.csv` | `session_id` | one session |
| `prompts.csv` | `prompt_id` | one real human prompt (pseudo-prompts excluded) |
| `prompts_text.csv` | `prompt_id` | full prompt text (omitted with `--no-text`) |
| `tokens.csv` | `prompt_id × model × token_type × is_sidechain` | **raw counts only**, no costs |
| `requests.csv` | `prompt_id × request_index` | one deduplicated API request (always written, V10) |
| `token_types.csv` | `token_type` | reference: machine key, label, description |
| `categories.csv` | `prompt_id` | category, complexity, classifier model (authored by `categorize`) |
| `quota_log.csv` | `snapshot_at × field` | append-only quota time series |

Notable contract points:

- `tokens.csv` carries `session_id` and `model` denormalized so usage attached to
  **pseudo-prompts** (`<session>:_continuation`, which have no `prompts.csv` row)
  stays attributable and priceable per model.
- `is_sidechain` (0/1) splits **subagent** usage out as a dimension of
  `tokens.csv`; summing over it reproduces the per-prompt totals (the **V7
  invariant**: any schema evolution keeps per-prompt sums identical and ends
  with a ccusage re-reconciliation).
- `requests.csv` is the **request grain** (one row per deduplicated API
  request): `session_id`, `prompt_id`, 1-based chronological `request_index`
  within the prompt, `timestamp`, `model`, `stop_reason`, `is_sidechain`,
  `post_compact`, and pivoted counts (`input_tokens` … `cache_write_1h_tokens`,
  `server_tool_use_requests`). Per prompt, its column sums equal the
  `tokens.csv` totals exactly (V7, tested); requests with no usage at all
  (synthetic notices) are skipped. `post_compact` (0/1) marks requests
  descending from the synthetic post-compaction continuation, until the next
  real human prompt.
- `extract_meta.json` records when the extract ran and its `--since`/`--until`
  **window** (`null` for a full extract). Readers (`load_dataset`,
  `dataset_from_csvs`, hence the CLI `Source:` line and the dashboard) append
  `window: since …` so a partial export is never presented as the full
  history.
- `token_type` is a machine key (`input`, `output`, `cache_read`,
  `cache_write_5m`, `cache_write_1h`, `server_tool_use`); human labels live in
  `token_types.csv`, never in the join key.
- `categories.csv` is the one CSV `extract` never touches — it is owned by
  `categorize` and joined at read time. Each row records its `classifier_model`
  (`heuristic-v2`, or an LLM model id). A heuristic run **re-classifies rows
  stamped by an older heuristic version** (so a classifier upgrade propagates)
  but never overwrites an LLM-classified row; `--llm` re-classifies nothing
  (no repeat API spend). The eleven categories are `plan`, `implementation`,
  `debug`, `refactor`, `review`, `test`, `docs`, `ops`, `question`,
  `followup`, `other`; ties are broken by a fixed category order, so the result
  does not depend on the process hash seed.

`export --flat` is a derived, denormalized view (one row per prompt, token
columns pivoted, a `cost_<provider>_usd` per provider) for spreadsheets/BI; it is
regenerated from the dataset, not a source of truth.

## JSONL format notes (by version)

The tool parses an **undocumented, evolving** format and pins parsing against
fixtures per Claude Code version under `tests/fixtures/claude-code-<version>/`
(capture new ones with `scripts/capture_fixture.py`). Observed behavior:

- **Progressive snapshots.** A single assistant message is written as several
  JSONL lines, each a growing snapshot of the same response: `output_tokens`
  increases line by line, and the `model` can change mid-message. Rule
  (determined by probing against ccusage): **largest usage snapshot wins, ties
  broken by the first line** (a message straddling midnight belongs to its start
  day). "First occurrence wins" under-counted output by ~2%.
- **Dedup key.** `message.id + requestId`, deduplicated **globally across all
  files** — resumed / `--resume` sessions replay the same records into a new
  file, so per-session dedup would double-count them.
- **Attribution drift (≈ 2.1.1xx and later).** Assistant events **no longer carry
  a `promptId`**. Attribution now walks the `uuid` / `parentUuid` chain back to a
  user prompt; unattributable tails (e.g. a resumed conversation replayed before
  its first prompt) are gathered under a `<session>:_continuation` pseudo-prompt
  — counted in costs, absent from `prompts.csv`.
- **Sub-agents.** Files under `subagents/` carry the **parent** `sessionId` and
  the parent prompt's id; their cost is rolled into the parent prompt and
  excluded from the parent's `assistant_turns`/`tool_calls`.
- **Cache granularity.** `usage.cache_creation.ephemeral_5m_input_tokens` /
  `ephemeral_1h_input_tokens` map to `cache_write_5m` / `cache_write_1h`; when
  only a total is present it falls back to `cache_write_5m`.
- **Filtered entries.** `isMeta: true`, `<command-name>` /
  `<local-command-stdout>` blocks, `[Request interrupted by user]`, and the
  synthetic post-compaction "This session is being continued…" message are kept
  out of `prompts.csv` and categorization; their token usage is still counted
  against the session.
- **Compaction marking.** The `isCompactSummary` continuation (or its
  "This session is being continued…" text form) flags its `uuid`; the flag is
  inherited down the `parentUuid` chain and reset by the next real human
  prompt — that is the `post_compact` column of `requests.csv`.

The extraction report (always printed by `extract`, gated to errors by
`--strict`) lists files read/skipped, unknown event types with counts, unpriced
models, and the Claude Code versions seen — so a silent upstream format change
surfaces loudly instead of corrupting totals. `scripts/reconcile_ccusage.py`
cross-checks day × model totals against `bunx ccusage --json`.
