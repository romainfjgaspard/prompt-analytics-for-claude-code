# Prompt-level analytics for Claude Code

> **Unofficial — not affiliated with Anthropic.** This is a community tool that reads Claude Code's local log files. "Claude" and "Claude Code" are trademarks of Anthropic.

[![CI](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/actions/workflows/ci.yml/badge.svg)](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/prompt-analytics-for-claude-code)](https://pypi.org/project/prompt-analytics-for-claude-code/)
[![Python](https://img.shields.io/pypi/pyversions/prompt-analytics-for-claude-code)](https://pypi.org/project/prompt-analytics-for-claude-code/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/live-demo-brightgreen.svg)](https://prompt-analytics-demo.streamlit.app)

Other tools tell you what you spent **per day** or **per session**. This one goes down to the **prompt**: every prompt you have ever sent becomes one row, priced from the raw token counts in your local `~/.claude/projects/**/*.jsonl` logs — no account, no API key, nothing leaves your machine.

- **Tokens and cost per prompt and per project.** A real dataset, not a daily total: which prompts and which projects actually cost you money, with a Pareto view of where it concentrates.
- **Meta-analyses across long sessions.** How the marginal cost of a prompt changes with its depth in the session, how the cache read/write mix shifts — the questions you can only ask once you have per-prompt rows.
- **Automatic categorization of your prompts.** A local, zero-dependency heuristic labels every prompt across thirteen categories (plan / implementation / debug / refactor / review / test / docs / ops / question / followup / feedback / notification / other) and scores its observed complexity 1–5. An offline **semantic** classifier (multilingual embeddings, still no API key) and an optional LLM pass can refine it (opt-in — see [Categorization](#categorization)).

It validates its token totals against [ccusage](https://github.com/ryoppippi/ccusage) (see [How the numbers stay accurate](#how-the-numbers-stay-accurate)). Think of it as the per-prompt dataset layer that sits alongside the report-style tools.

> 📖 **Full documentation is in the [project wiki](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki)** — [Architecture](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Architecture), a [Codebase Guide](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Codebase-Guide), the complete [CLI reference](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/CLI-Commands), the [Dashboard](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Dashboard) pages, and [how the numbers stay accurate](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Accuracy-and-Pricing). This README is the quick tour.

## See it without installing anything

**Two surfaces on the same data — a Streamlit dashboard and a terminal CLI.**

The richest way in is the **dashboard**, and a live demo runs it on a synthetic dataset (no real prompts), so you can explore every page before installing — **the image below links to the live demo:**

**▶ Live demo: [prompt-analytics-demo.streamlit.app](https://prompt-analytics-demo.streamlit.app)**

[![The prompt-analytics Streamlit dashboard — KPI cards, cost by token type, and daily cost by model](https://raw.githubusercontent.com/romainfjgaspard/prompt-analytics-for-claude-code/main/docs/screenshots/dashboard-home.png)](https://prompt-analytics-demo.streamlit.app)

The same numbers are one line away in the **terminal**. For example, `by-category` auto-labels each prompt and scores its *observed* complexity 1–5 — a local heuristic, no LLM required:

![`prompt-analytics by-category` in the terminal](https://raw.githubusercontent.com/romainfjgaspard/prompt-analytics-for-claude-code/main/docs/screenshots/cli-by-category.png)

## Quick start

The richest way in is the **dashboard** — install with the `dashboard` extra and point it at your own data:

```bash
uv tool install "prompt-analytics-for-claude-code[dashboard]"
prompt-analytics dashboard          # refresh your data, then launch Streamlit on http://localhost:8501
```

`dashboard` refreshes the data first (extract + snapshot + local categorize, no API key) and then opens the board — so a fresh launch never shows stale numbers. Pass `--no-refresh` to skip that and open on the existing CSVs. (Plain pip: `pip install "prompt-analytics-for-claude-code[dashboard]"`.)

No data yet, or just curious? The [**live demo**](https://prompt-analytics-demo.streamlit.app) runs the same dashboard on synthetic data — a tour of every page is in the wiki ([Dashboard](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Dashboard)).

### Prefer the terminal? The same data as a CLI

Every analysis is also a single line over your local logs — zero install with [uv](https://docs.astral.sh/uv/):

```bash
uvx --from prompt-analytics-for-claude-code prompt-analytics summary
```

Use it often? `uv tool install prompt-analytics-for-claude-code` (or `pip install prompt-analytics-for-claude-code`), then `prompt-analytics summary`.

The summary is a live parse of your logs — no `extract` step required (that one is just an export, see below). Running on the bundled demo dataset, the output looks like this; on your machine the source line reads `live parse of ~/.claude/projects`:

```text
Usage summary
+------------------------------------------------------------------+
| Metric                     | Value                               |
|----------------------------+-------------------------------------|
| Sessions                   | 80                                  |
| Prompts                    | 798                                 |
| Projects                   | 5                                   |
| Period                     | 2025-12-17 .. 2026-06-09 (175 days) |
| Input tokens               | 1,729,726                           |
| Output tokens              | 2,961,653                           |
| Cache read tokens          | 497,973,616                         |
| Cache write (5m) tokens    | 22,683,667                          |
| Cache write (1h) tokens    | 4,680,558                           |
| Server tool use (requests) | 264                                 |
| Total tokens               | 530,029,220                         |
| Cost (anthropic)           | $431.85                             |
| Cost (copilot)             | $413.84                             |
| Subagents                  | $6.19 (1.4% of anthropic cost)      |
+------------------------------------------------------------------+
  Source: demo_data CSVs.
```

**Prerequisites:** Python 3.10+ and Claude Code with at least one recorded session under `~/.claude/projects/`. No organization or Admin API key required (unlike the official Claude Code Analytics API, which needs both).

## What it can tell you

Per-prompt rows answer questions a daily total can't. Every block below is real output from the bundled demo dataset (`--from-csv demo_data`); on your machine the source line reads `live parse of ~/.claude/projects`.

### What one prompt actually costs, by session depth

The view report-style tools don't give: cost **per prompt**, bucketed by its position in the session. The mechanism is context rent — at depth 21–50 the **median request re-reads ×2.71 the depth-1 context (≈344k tokens)**, paid on *every* turn until `/compact` resets it (`context`). `sessions --depth` and the dashboard box plot then show how that translates into the per-prompt cost distribution.

### Where the bill goes: context rent

Almost none of your spend is the text you type or the model writes — **83% of the bill is "context rent"**, the cache reads and writes that re-send the conversation every turn (`by-token-type`: cache reads alone are 94% of *tokens* but 46% of *cost*). The same numbers are now also available **per prompt**.

### The hidden cache costs: TTL expiry, compaction, overhead

Cache entries expire (5m / 1h TTL); pause longer and the next turn pays to **re-write** the whole context. `ttl` blames those re-writes on the pause that caused them (≈$36 here). `compactions` does the mirror for `/compact` (context before/after + the rebuild cost), and `overhead` isolates the fixed per-session cost (system prompt + CLAUDE.md + MCP tools).

### Subagents, billed back to the prompt that spawned them

Sub-agent (`isSidechain`) work is parsed and rolled into the parent prompt, so nothing is under-counted — `summary` calls it out (`Subagents: $6.19`) and `by-model` adds a per-model column.

### Is a Pro/Max plan worth it vs the API?

`break-even` prices your whole history on the per-token grid, projects it to a month, and compares each flat-rate plan:

```text
Plan break-even: API-equivalent vs subscription (anthropic)
+--------------------------------------------------------------------+
| Plan           | Plan $/mo | Your API $/mo | vs plan | Saving $/mo |
|----------------+-----------+---------------+---------+-------------|
| Claude Pro     |    $20.00 |        $74.03 |   x3.70 |      $54.03 |
| Claude Max 5x  |   $100.00 |        $74.03 |   x0.74 |     $-25.97 |
| Claude Max 20x |   $200.00 |        $74.03 |   x0.37 |    $-125.97 |
+--------------------------------------------------------------------+
  Source: demo_data CSVs.
  Over the 175-day window your usage is worth $431.85 of anthropic API
  ($74.03/month projected at this rate).
  At this rate the Claude Pro plan ($20.00/mo) pays for itself: your
  API-equivalent is $74.03/month (x3.7 the price), an estimated $54.03/month
  cheaper than paying per token.
  Quota windows (peak utilization seen via `snapshot`): five_hour 100%,
  seven_day 84%, seven_day_sonnet 39%. High utilization means you are
  extracting most of the plan's allowance; low utilization means headroom
  (or a smaller plan would do).
```

## Commands

Every analysis command works **on the fly** — it parses your JSONL in memory (~0.5 s) when there is no fresh `extract` to read, so you never have to run `extract` first. All of them accept `--output-dir DIR`, `--format table|csv|json`, `--pricing PATH`, `--no-cache`, `--from-csv DIR` (analyze the CSVs in DIR as-is, with no live parse and no freshness check — the CLI counterpart of what the dashboard does, e.g. `summary --from-csv demo_data`), and `--since YYYY-MM-DD` / `--until YYYY-MM-DD` to restrict the analysis to a date range (inclusive, e.g. `summary --since 2026-06-01`).

Commands that price tokens take `--provider NAME` to choose the rate card for the cost column (`NAME` is a provider key from the pricing grid — `anthropic` (default) or `copilot` ship by default; see [Pricing providers](#pricing-providers)).

**Totals and breakdowns**

| Command | Key flags | What it shows |
| --- | --- | --- |
| `summary` | | Sessions, prompts, tokens by type, cost per provider, period, subagent share. |
| `by-project` | `--provider NAME` | Cost / tokens / prompts per project, sorted by cost, with each project's share and the running cumulative %. |
| `by-model` | `--compact` `--provider NAME` | Token and cost split per model (cache writes split by TTL, subagent column); `--compact` fits 80 columns. |
| `by-token-type` | `--provider NAME` | Cost split per token type — the **context-rent** share of the bill (cache vs generation vs input). |
| `by-category` | `--provider NAME` | Cost and observed complexity per category (needs `categorize`). |
| `timeline` | `--by day\|week\|month` `--provider NAME` | Cost / prompts / tokens grouped by calendar period (chronological), each with its share of the total. |
| `prompts` | `--top N` `--provider NAME` | The N most expensive prompts, with a preview. |
| `sessions` | `--depth` \| `--top N`, `--project NAME` | Sessions ranked by cost, or `--depth` for the marginal-cost-by-depth meta-analysis; `--project` restricts to one. |

**Cost by content** — the same reconciled bill, attributed across four levels of *content* (input → output → context → task), each metrics-only (no source code is ever stored):

| Command | Shows |
| --- | --- |
| `by-output` | **Output** composition: language mix (by extension), code vs tests, lines +/−, and the prose-vs-code split of generation cost. |
| `by-context` | **Context** composition: what fills the re-read cache by source, splitting one-off **loading** (`cache_write`) from **rent** (`cache_read` paid every turn it lingers). |
| `by-file` | Per-**file** footprint: edits + line diff (output) crossed with reads + context cost — the actionable "what to keep out of context". |
| `by-task` | Cost by **task** (the unit of work, not the prompt): total cost with context share, prompts, span, dominant category. Built from the `TodoWrite` spine with an inference fallback. |
| `impact` | **Before/after** a `--pivot YYYY-MM-DD` date: workload-normalized ratios (cost/prompt, output share, context rent share) with the workload confounders alongside — an observational split, not a controlled experiment. |

These power the dashboard's **Composition** page ("where your cost goes, by content") and its dedicated before/after **Compare** tab.

Beyond these, there are **power-user analyses** on the request grain (`context`, `ttl`, `compactions`, `overhead`, `model-category`, `recommend`, `burn-rate`, `break-even`) and the **pricing / export / pipeline** commands (`compare`, `export`, `extract`, `snapshot`, `categorize`, `run`, `dashboard`, `config init`). The full reference, grouped by purpose with every flag, is in the wiki:

- 📖 [**CLI Commands** — power-user analyses](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/CLI-Commands#power-user-analyses-request-grain)
- 📖 [**CLI Commands** — pricing, export & pipeline](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/CLI-Commands#pricing-export-and-pipeline)

Run `prompt-analytics <command> --help` for the full flag list. `extract` and `run` always regenerate the whole output atomically — there is no incremental mode to misuse, and changing the pricing never requires a re-extract (costs are computed at read time from raw counts).

## Exporting data

`extract` writes the canonical, inspectable, zero-dependency format: relational CSVs keyed by `prompt_id` / `session_id` (`sessions`, `prompts`, `prompts_text`, `tokens`, `requests`, `token_types`, plus `quota_log` from `snapshot`) into `./output` (gitignored). `tokens.csv` holds **raw counts only** — prices are computed at read time, so a pricing change never needs a re-extract. The exact column layout is the single source of truth in [`prompt_analytics/schema.py`](prompt_analytics/schema.py); the full data flow and CSV contract are in the wiki ([Architecture → the data model](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Architecture#the-data-model-csv-contract)).

For spreadsheets and BI tools that want one flat table, `export --flat` writes a single denormalized CSV — one row per prompt, token columns pivoted (`input_tokens`, `output_tokens`, `cache_read_tokens`, …), a `cost_<provider>_usd` column per provider, and the session fields duplicated in:

```bash
prompt-analytics export --flat --out my_usage.csv
```

Every command also takes `--format json` (a `{title, rows, notes}` object with raw numeric values) and `--format csv` (raw rows on stdout, notes on stderr) for piping into `jq`, pandas, or a notebook.

## Pricing providers

Costs are computed at read time from a generic multi-provider grid ([`prompt_analytics/data/pricing.yml`](prompt_analytics/data/pricing.yml)). Two ship by default — **`anthropic`** (published API rates) and **`copilot`** (the GitHub Copilot equivalent) — so `compare` answers *"what would this usage cost billed through Copilot instead of the API?"*. Add any rate card under `providers:` (an internal plan, a Bedrock tier, a negotiated rate) or pass `--pricing ./my.yml`; the bundled grid is kept honest by a weekly CI drift job (LiteLLM + the live Copilot page). See [CONTRIBUTING.md](CONTRIBUTING.md) for the schema, or the wiki for [how pricing works](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Accuracy-and-Pricing#how-pricing-works) (lookup rules, drift job).

<details>
<summary><strong>Example: <code>compare</code> across providers</strong></summary>

```text
Provider cost comparison
+---------------------------------------------------------------------+
| Model             |      Tokens | Cost (anthropic) | Cost (copilot) |
|-------------------+-------------+------------------+----------------|
| claude-opus-4-8   | 205,870,608 |          $211.45 |        $202.49 |
| claude-fable-5    |  53,609,993 |          $113.85 |        $109.50 |
| claude-sonnet-4-6 | 144,545,831 |           $82.41 |         $79.14 |
| claude-haiku-4-5  | 126,002,788 |           $24.13 |         $22.71 |
| TOTAL             | 530,029,220 |          $431.85 |        $413.84 |
+---------------------------------------------------------------------+
  Total on copilot: x0.96 the anthropic total.
```

</details>

## Categorization

By default, categorization is **100% local and runs for everyone** — no API key, no network:

```bash
prompt-analytics categorize        # heuristic, writes categories.csv
prompt-analytics by-category
```

A weighted FR+EN regex classifier labels each prompt across thirteen categories (plan / implementation / debug / refactor / review / test / docs / ops / question / followup / feedback / notification / other), and an **observed** complexity 1–5 is derived from the effort it actually triggered (turns, tool calls, length, cost) — a measurement, not a guess. Read `$/prompt (med)` alongside the total: a category can top the bill on volume alone, but a high median per prompt flags work that is *intrinsically* expensive (here `debug` is costly both ways at $0.73/prompt, while `followup` stays cheap at $0.26).

<details>
<summary><strong>Example: <code>by-category</code> output</strong></summary>

```text
Cost by category
+---------------------------------------------------------------------------------------------------+
| Category       | Prompts | Prompt % | Avg complexity | $/prompt (med) | Cost (anthropic) | Cost % |
|----------------+---------+----------+----------------+----------------+------------------+--------|
| debug          |     138 |    17.3% |              4 |          $0.73 |          $113.03 |  26.2% |
| plan           |      78 |     9.8% |              3 |          $0.56 |           $60.35 |  14.0% |
| implementation |     108 |    13.5% |            3.5 |          $0.40 |           $54.32 |  12.6% |
| question       |     105 |    13.2% |              2 |          $0.25 |           $39.62 |   9.2% |
| ops            |      71 |     8.9% |            1.9 |          $0.41 |           $36.28 |   8.4% |
| other          |      85 |    10.7% |              2 |          $0.21 |           $31.08 |   7.2% |
| docs           |      57 |     7.1% |            2.1 |          $0.35 |           $25.60 |   5.9% |
| review         |      49 |     6.1% |            3.5 |          $0.39 |           $24.99 |   5.8% |
| test           |      56 |     7.0% |            2.7 |          $0.27 |           $20.92 |   4.9% |
| refactor       |      21 |     2.6% |              4 |          $0.46 |           $12.43 |   2.9% |
| followup       |      30 |     3.8% |            1.3 |          $0.26 |           $12.06 |   2.8% |
+---------------------------------------------------------------------------------------------------+
  Source: demo_data CSVs.
```

`Prompt %` and `Cost %` sit next to their own columns, so volume and spend are never conflated.

</details>

<details>
<summary><strong>Offline semantic refinement (no API key)</strong></summary>

`categorize --semantic` adds a third, **fully offline** classifier between the heuristic and the LLM: it reads *meaning* via multilingual embeddings, so it places FR+EN prompts the regex can't and trims the `other` long tail — with **no API key and nothing leaving your machine**.

```bash
prompt-analytics categorize --semantic   # offline embeddings, writes categories.csv
```

The embeddings ship **in the core package** — there is no heavy extra to install. It uses a static, torch-free [`model2vec`](https://github.com/MinishLab/model2vec) model (a few tens of MB, pure-numpy at inference, fetched once then cached for offline use). It is **mono-label**: each prompt scores the cosine similarity to per-category prototype examples (FR+EN), fused with a light lexical prime for `ops`/`feedback`; the top score wins, or falls back to `other` below a threshold τ.

It is **opt-in on purpose** — the heuristic stays the default. A litmus + LLM-judge evaluation on the demo corpus (`scripts/eval_semantic.py`) had the heuristic edge it out (80% vs 72% litmus), partly because synthetic, keyword-rich prompts favor the regex and human agreement on prompt intent tops out around ~0.7. The semantic mode is most useful on real, multilingual, open-ended prompts. Like the LLM, it only **supersedes heuristic** rows, never LLM-classified ones.

Power users can tune the calibrated defaults (`τ`, lexical prime weight, prototype top-k) via `--tau` / `--prime-weight` / `--top-k`, or a `semantic:` section in `config.yml` (CLI flag > config > calibrated default).

`categorize --audit-categories` is a related diagnostic: it clusters the **whole corpus** (HDBSCAN) and compares the natural clusters to the thirteen categories — flagging candidate merges/splits and the `other` bucket. It only writes a report + CSV; it never changes `categories.csv`.

</details>

<details>
<summary><strong>Optional LLM refinement</strong></summary>

`categorize --llm` re-labels prompts with an LLM; it only overwrites heuristic rows, never the reverse. Providers:

- `--provider anthropic` (default when `ANTHROPIC_API_KEY` is set; `--batch` uses the Message Batches API, −50% cost).
- `--provider openrouter` — **a third party.** Up to ~2000 characters of each prompt are sent to OpenRouter; the command warns you at runtime before sending.
- `--provider ollama` — a **local** model via the OpenAI-compatible API at `localhost:11434/v1`. Free, no key, nothing leaves your machine.
- `--provider azure` — **a third party** (Azure OpenAI); like OpenRouter, prompt excerpts are sent to it.

Set keys in a `.env` file (see [`.env.example`](.env.example)). Most Claude Code Pro/Max subscribers have no Console API key — that is exactly why the heuristic is the default.

</details>

## How the numbers stay accurate

Claude Code writes a single assistant message as **several JSONL lines carrying progressive snapshots** of the same response — `output_tokens` grows line by line, and the model can even change mid-message. Naively summing every line double-counts and inflates token totals by roughly 2.5×.

The parser counts each response exactly once: for a given `message.id + requestId` it keeps the **largest** usage snapshot (ties broken by the first line, so a message straddling midnight belongs to its start day). Deduplication is **global across all files**, not per session — that is what corrects resumed / `--resume` sessions, where the same records are replayed into a new file.

On top of dedup, the extractor filters fake prompts (`isMeta`, `<command-*>` blocks, interruptions, the post-compaction notice — kept out of `prompts.csv` but still counted against the session), rolls **sub-agent** (`isSidechain`) cost into the parent prompt, and splits cache writes by TTL (`cache_write_5m` 1.25× vs `cache_write_1h` 2×).

These totals reconcile **bucket-for-bucket with `bunx ccusage --json`** (day × model) on real history — the reconciliation is scripted in `scripts/reconcile_ccusage.py`. Each command prints its data source, and `extract` ends with a loud report so a silent format break surfaces immediately.

- 📖 Full detail: [**Accuracy and Pricing**](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Accuracy-and-Pricing) — [why the counts are right](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Accuracy-and-Pricing#why-the-token-counts-are-right), [ccusage reconciliation](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Accuracy-and-Pricing#reconciliation-against-ccusage), and the [V7 invariant](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Accuracy-and-Pricing#the-v7-invariant).

## Privacy

This tool reads your entire Claude Code prompt history, so it is explicit about what it touches.

**Reads (locally):** `~/.claude/projects/**/*.jsonl` (read-only); `~/.claude/.credentials.json` (only `snapshot`, to reuse your existing token); the pricing grid bundled in the package. A relocated Claude directory is honored via `CLAUDE_CONFIG_DIR`, like Claude Code itself. On Windows, `~` means `C:\Users\<you>` (`%USERPROFILE%`).

**Writes (locally, never inside the installed package):** the CSVs in your output directory (default `./output`, gitignored). `prompts_text.csv` contains the **full text of every prompt in clear**, and `prompts.csv` keeps a short `prompt_preview` — treat an export as sensitive before sharing or committing it. A parse cache lives under `%LOCALAPPDATA%\prompt-analytics\` on Windows, `$XDG_CACHE_HOME/prompt-analytics/` (default `~/.cache/prompt-analytics/`) elsewhere — safe to delete; entries for deleted logs are garbage-collected on each run.

- `--no-text` is honest: it skips `prompts_text.csv` **and** blanks the `prompt_preview` column.
- Free-text cells starting with `=`, `+`, `-`, or `@` are prefixed with `'` so a crafted prompt can't execute as a spreadsheet formula.

**Network:** **nothing by default.** `extract` and every analysis command are fully local. `snapshot` reuses the token Claude Code stored in `~/.claude/.credentials.json` to call an **undocumented** Anthropic OAuth usage endpoint (sent to Anthropic only; it fails gracefully if the endpoint changes). `categorize --llm` sends prompt excerpts to the provider you choose — and OpenRouter is a third party.

## Related tools

- **[ccusage](https://github.com/ryoppippi/ccusage)** — fast terminal usage reports (daily/session/blocks) and JSON output, 14+ CLIs supported. Complementary: it does aggregated reports, this does the per-prompt dataset. Token totals here are validated against it.
- **[usage-monitor-for-claude](https://github.com/jens-duttke/usage-monitor-for-claude)** — real-time quota tray app for Windows; it discovered the undocumented quota endpoint this tool's `snapshot` reuses.
- **[Claude-Code-Usage-Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)** — real-time 5-hour-block burn-rate monitor with P90 predictions.
- **Claude Code Analytics API** — official, org-level analytics; requires an organization and an Admin API key.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, how to add a pricing provider, and how to capture a versioned test fixture. In short: `uv sync --all-extras`, then `ruff check .`, `ruff format --check .`, `mypy prompt_analytics tests`, and `pytest`.

## License

[MIT](LICENSE)
