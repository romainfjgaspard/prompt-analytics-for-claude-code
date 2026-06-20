# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versioning
note: the upstream Claude Code JSONL format is unstable, so parsing breakage is
treated as expected and reflected in patch/minor bumps; `2.0.0` marks the
**cost-by-content** milestone (four levels of cost attribution), not a format
break.

## [Unreleased]

## [2.0.0] — 2026-06-20 — Cost by content

Where your cost goes, **by content** — four levels of attribution on the same
reconciled bill: **input** (category), **output** (what the assistant produced),
**context** (what fills the re-read cache), and **task** (the unit of work). Plus
a transverse **before/after** mode to measure the impact of an optimization.
Every new number reconciles to the bill by construction, stays **metrics-only**
(no source code persisted), and is exposed on both surfaces (CLI + dashboard).

### Added — output composition (Axe C)
- `by-output`: language mix (by extension), code vs tests, lines added/deleted
  (exact LCS diff), and the **prose-vs-code split** of output tokens/cost (priced
  via a local tokenizer, `tiktoken` at the core with an offline fallback).
- `extract` now derives output metrics into `output_files.csv` (per `prompt_id` ×
  path) and `output_tokens.csv` (prose/code token split) — metrics only, never
  the diff content.

### Added — context composition (Axe D)
- `by-context`: what fills the cached, re-read context, by source (config / files
  read by language / tool output / conversation) — splitting **loading**
  (`cache_write`, one-off) from **rent** (`cache_read` paid every turn it
  lingers). The metric is *size × turns present*; totals reconcile to the
  main-chain cache bill (with an honest `(unattributed)` bucket).
- `by-file`: a per-file footprint crossing edits + line diff (output) with reads
  + context cost (loading + rent) — the actionable "what to keep out of context".
- New `context_sources.csv` and `context_cost.csv` (metrics only; relative paths
  as file identity, never absolute paths or content).

### Added — semantic categorization & taxonomy audit (Axe B1)
- `categorize --semantic`: an offline, mono-label semantic classifier built on
  multilingual static embeddings (`model2vec`, torch-free, shipped in the core
  package — no extra to install, no API key, nothing leaves your machine). It
  fuses cosine similarity to per-category FR+EN prototypes with a lexical prime,
  and falls back to `other` below a calibrated threshold. **Opt-in**: the
  heuristic stays the default (it edged out the semantic mode on the demo
  litmus/LLM-judge evaluation). Tunable via `--tau` / `--prime-weight` /
  `--top-k` or a `semantic:` section in `config.yml`.
- `categorize --audit-categories`: a diagnostic that clusters the whole corpus
  (HDBSCAN) and compares the natural clusters to the thirteen categories
  (alignment matrix, c-TF-IDF labels, candidate merges/splits). Writes a report
  + CSV only; never changes `categories.csv`.
- `categorize --llm --provider azure`: Azure OpenAI as an LLM provider for the
  optional refinement pass.

### Added — task attribution (Axe B2)
- `by-task`: cost by **task** (the unit of work, not the prompt) — total cost
  with its context share, prompt count, span, and dominant category. Tasks are
  built from the real `TodoWrite` spine in the transcript, with an inference
  fallback (time gaps + embeddings + category structure) when there are no todos.
  Cost reconciles to the bill by construction (each prompt belongs to one task);
  session overhead is excluded and noted.
- `extract` now captures `TodoWrite` into `tasks.csv` and `task_prompts.csv`
  (task names from Claude's own todo labels; otherwise metrics only).
- Dashboard: a **task cost graph** (force-layout: tasks as nodes sized by cost
  and colored by dominant category, prompts as satellites) in the Composition
  page.

### Added — before/after impact mode (Axe E)
- `impact --pivot YYYY-MM-DD`: split the history on a switch date and show
  **workload-normalized ratios** (cost per prompt, output cost share, context
  rent share, cache read per turn, output tokens per prompt) with the workload
  **confounders** (volume, depth, task mix) alongside — an observational split,
  not a controlled experiment, and labelled as such. Without `--pivot`, lists
  detected config-change dates (mtime of CLAUDE.md / settings.json) as a hint.
- Dashboard: a global **"Compare before/after a change"** sidebar mode — when on,
  the Overview and Composition pages reframe as before vs after + grey deltas
  (built from the same `impact` report, so the table and cards never drift).

### Added — dashboard & misc
- New **Composition** dashboard page narrating "where your cost goes, by content"
  (input → output → context → files → tasks), plus the global date-pivot mode.
- `--since YYYY-MM-DD` / `--until YYYY-MM-DD` on every analysis command (inclusive
  date range, e.g. `summary --since 2026-06-01`).
- `timeline` command: cost / prompts / tokens grouped by `--by day|week|month`.

### Changed
- The local heuristic now labels **thirteen** categories (added `feedback` and
  `notification`, trimming the `other` long tail).
- `by-project` now always shows the cumulative % column; the `--pareto` flag is
  deprecated (still accepted as a no-op so existing invocations keep working).
- `--provider` now documents its expected value (`--provider NAME`) and lists the
  known providers (`anthropic`, `copilot`) in `--help`.

## [0.3.1] — 2026-06-15

### Fixed
- README screenshots and the PyPI / Python badges now use absolute URLs, so they
  render on the PyPI project page (relative paths only resolve on GitHub).

## [0.3.0] — 2026-06-15 — Initial public release

Prompt-level analytics for Claude Code: every prompt in your local
`~/.claude/projects/**/*.jsonl` logs becomes one priced row — no account, no API
key, nothing leaves your machine. Two surfaces on the same data: a terminal CLI
and a Streamlit dashboard.

### Highlights
- **Per-prompt dataset** — tokens and cost per prompt, project, model, category
  and session, with a Pareto view of where the spend concentrates.
- **Power-user analyses at the request grain** — `context` (accumulated context
  by session depth), `ttl` (cache-TTL expiry losses), `compactions`, `overhead`,
  `by-token-type` (the **context-rent** share of the bill), `model-category
  --whatif`, `recommend`, `burn-rate`, and `break-even` (is a Pro/Max plan worth
  it vs the API?).
- **Automatic categorization** — a local, zero-dependency FR+EN heuristic labels
  every prompt across eleven categories and scores its *observed* complexity 1–5;
  an optional LLM pass (Anthropic / OpenRouter / local Ollama) refines it.
- **Streamlit dashboard** — Apache ECharts on a dark-by-default theme, global
  cross-filtering (click to filter, brush a date range), an Explorer drill-down
  (day → session → prompt), and a public synthetic-data demo.
- **Accurate by construction** — global cross-file deduplication on
  `message.id + requestId` (fixes the ~2.5× token inflation and double-counted
  resumed / `--resume` sessions), fake-prompt filtering, an explicit subagent
  policy, and cache writes split by TTL (5m vs 1h). Totals reconcile
  bucket-for-bucket with [ccusage](https://github.com/ryoppippi/ccusage) on real
  history.
- **Generic multi-provider pricing** — `pricing.yml` ships `anthropic` and
  `copilot` grids and accepts any rate card; costs are computed at read time from
  raw counts, so a pricing change never needs a re-extract.
- **Inspectable exports** — relational CSVs keyed by `prompt_id` / `session_id`,
  `--format table|csv|json` on every command, and `export --flat` for Excel/BI.
- **`snapshot`** — records plan quota utilization over time via the OAuth usage
  endpoint Claude Code already uses (kept out of any public API; fails
  gracefully).

### Privacy
Fully local by default: `extract` and every analysis command touch only your
local logs. `snapshot` calls Anthropic's own OAuth usage endpoint with your
existing token; `categorize --llm` sends prompt excerpts only to the provider you
choose (OpenRouter is a third party). See the README's Privacy section for the
full read/write/network breakdown.
