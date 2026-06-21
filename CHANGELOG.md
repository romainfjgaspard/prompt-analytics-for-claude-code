# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versioning
note: pre-1.0, new features land as **minor** bumps and the upstream Claude Code
JSONL format is unstable, so parsing breakage is treated as expected and reflected
in patch/minor bumps.

## [Unreleased]

## [0.4.0] — 2026-06-22 — Cost by content

Where your cost goes, **by content** — the same reconciled bill attributed across
the layers of *content* it pays for: **input** (what you ask, by category),
**output** (what the assistant writes), **context** (what fills the re-read cache),
and the **files** and **tasks** the work touches. Plus a before/after view to
measure the impact of a change. Every new number reconciles to the bill by
construction and stays **metrics-only** — no source code is ever persisted. Exposed
on both surfaces: new CLI commands and a rebuilt dashboard.

### Added — cost-by-content commands
- `by-output`: output composition — language mix (by extension), code vs tests,
  lines added/deleted (exact diff), and the **prose-vs-code split** of generation
  tokens/cost (priced via a local tokenizer with an offline fallback).
- `by-context`: what fills the cached, re-read context, by source (config / files
  read by language / tool output / conversation) — splitting one-off **loading**
  (`cache_write`) from **rent** (`cache_read`, paid every turn it lingers). Totals
  reconcile to the main-chain cache bill (with an honest `(unattributed)` bucket).
- `by-file`: a per-file footprint crossing edits + line diff with reads + context
  cost (loading + rent) — the actionable "what to keep out of context".
- `by-task`: cost by **task** (the unit of work, not the prompt) — total cost with
  its context share, prompt count, span, and dominant category, built from the
  `TodoWrite` spine with an inference fallback when there are no todos. Cost
  reconciles to the bill by construction (each prompt belongs to one task); session
  overhead is excluded and noted.
- `impact --pivot YYYY-MM-DD`: split the history on a date and show
  **workload-normalized ratios** (cost per prompt, output cost share, context-rent
  share, cache-read per turn, output tokens per prompt) with the workload
  **confounders** (volume, depth, mix) alongside — an observational split, not a
  controlled experiment, and labelled as such. Without `--pivot`, lists detected
  config-change dates (mtime of CLAUDE.md / settings.json) as a hint.
- `timeline`: cost / prompts / tokens grouped by `--by day|week|month`.
- `extract` now derives the metrics behind these into new CSVs (`output_files`,
  `output_tokens`, `context_sources`, `context_cost`, `tasks`, `task_prompts`) —
  metrics only, relative paths as file identity, never content or absolute paths.

### Added — semantic categorization & taxonomy audit
- `categorize --semantic`: an offline, mono-label semantic classifier on
  multilingual static embeddings (`model2vec`, torch-free, shipped in the core
  package — no extra to install, no API key, nothing leaves your machine). It
  fuses cosine similarity to per-category FR+EN prototypes with a lexical prime,
  and falls back to `other` below a calibrated threshold. **Opt-in** in the CLI:
  the heuristic stays the default. Tunable via `--tau` / `--prime-weight` /
  `--top-k` or a `semantic:` section in `config.yml`.
- `categorize --audit-categories`: a diagnostic that clusters the whole corpus and
  compares the natural clusters to the categories (alignment matrix, labels,
  candidate merges/splits). Writes a report + CSV only; never changes
  `categories.csv`.
- `categorize --llm --provider azure`: Azure OpenAI as a provider for the optional
  LLM refinement pass.

### Added — dashboard
- A **Composition** page narrating "where your cost goes, by content" — four
  sections (Input · Output · Context · Files), each a band of **4 KPIs + 2 charts**.
  Input shows cost by category + a prompt-length histogram (both click-to-filter);
  Output a prose-vs-code donut + code language mix; Context a loading-vs-rent donut
  + a tokens Pareto; Files a force-layout **file cost-graph** (files sized by
  reconciled context cost, coloured by language, with edit-prompt satellites, a
  project picker and a per-file drill).
- The detail views split into a **Prompt Explorer** (session → prompt drill,
  cumulative-cost timeline, full prompt text) and a new **File Explorer** (the
  per-file footprint table with filters + a per-file drill: who edited it, which
  sessions kept it in context, and the load + rent it cost), with **deep-links**
  between them and from the charts.
- A dedicated **Compare** tab: pick a pivot date and read the five
  workload-normalized ratios as Before | After, plus two average-based charts
  (cost/prompt by token type, category mix) — averages and ratios only, never sums,
  reusing the `impact` numbers verbatim.
- **Global cross-filters** extended beyond model / project / category / date: a
  prompts-per-session bar (Sessions) and a prompt-length bucket (Composition) now
  filter the whole board too, and the category chart is clickable; an active filter
  offers Explore-prompts / Explore-files drill-throughs and a Reset.
- The in-app **Refresh** defaults to **semantic** categorization (offline, no API
  key), with a heuristic fallback when the model can't load.
- Removed the old **Prompts** tab (folded into Composition and the Prompt
  Explorer). Tab order: Home · Usage · Models · Sessions · Session depth ·
  Composition · Prompt Explorer · File Explorer · Optimize · Compare · Quotas · How
  it works.

### Changed
- The local heuristic now labels **thirteen** categories (added `feedback` and
  `notification`, trimming the `other` long tail).
- `--since YYYY-MM-DD` / `--until YYYY-MM-DD` on every analysis command (inclusive
  date range, e.g. `summary --since 2026-06-01`).
- `by-project` always shows the cumulative % column; the `--pareto` flag is
  deprecated (still accepted as a no-op so existing invocations keep working).
- `--provider` documents its expected value (`--provider NAME`) and lists the known
  providers (`anthropic`, `copilot`) in `--help`.

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
