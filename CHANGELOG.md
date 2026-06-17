# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — kept at
`0.x` on purpose: the upstream Claude Code JSONL format is unstable, so parsing
breakage is treated as expected and reflected in the version.

## [Unreleased]

### Added
- `--since YYYY-MM-DD` / `--until YYYY-MM-DD` on every analysis command (inclusive
  date range, e.g. `summary --since 2026-06-01`).
- `timeline` command: cost / prompts / tokens grouped by `--by day|week|month`.

### Changed
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
