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
- **Automatic categorization of your prompts.** A local, zero-dependency heuristic labels every prompt across eleven categories (plan / implementation / debug / refactor / review / test / docs / ops / question / followup / other) and scores its observed complexity 1–5. An optional LLM pass refines it (opt-in — see [Categorization](#categorization)).

It validates its token totals against [ccusage](https://github.com/ryoppippi/ccusage) (see [How the numbers stay accurate](#how-the-numbers-stay-accurate)). Think of it as the per-prompt dataset layer that sits alongside the report-style tools.

> 📖 **Full documentation is in the [project wiki](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki)** — [Architecture](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Architecture), a [Codebase Guide](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Codebase-Guide), the complete [CLI reference](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/CLI-Commands), the [Dashboard](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Dashboard) pages, and [how the numbers stay accurate](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Accuracy-and-Pricing). This README is the quick tour.

## See it without installing anything

**Two surfaces on the same data — a terminal CLI and a Streamlit dashboard.**

Every CLI command is a single line over your local logs. For example, `by-category` auto-labels each prompt and scores its *observed* complexity 1–5 — a local heuristic, no LLM required:

![`prompt-analytics by-category` in the terminal](https://raw.githubusercontent.com/romainfjgaspard/prompt-analytics-for-claude-code/main/docs/screenshots/cli-by-category.png)

And a live Streamlit demo runs on a synthetic dataset (no real prompts), so you can explore the dashboard before installing — **the image below links to the live demo:**

**▶ Live demo: [prompt-analytics-demo.streamlit.app](https://prompt-analytics-demo.streamlit.app)**

[![The prompt-analytics Streamlit dashboard — KPI cards, cost by token type, and daily cost by model](https://raw.githubusercontent.com/romainfjgaspard/prompt-analytics-for-claude-code/main/docs/screenshots/dashboard-home.png)](https://prompt-analytics-demo.streamlit.app)

## Quick start

Zero install, straight from PyPI with [uv](https://docs.astral.sh/uv/) — this reads your real `~/.claude/projects` and prints a summary in the terminal:

```bash
uvx --from prompt-analytics-for-claude-code prompt-analytics summary
```

Use it often? Install the `prompt-analytics` command on your PATH:

```bash
uv tool install prompt-analytics-for-claude-code
prompt-analytics summary
```

Or with pip: `pip install prompt-analytics-for-claude-code`, then `prompt-analytics summary`.

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

### Dashboard

Prefer a UI? Install with the `dashboard` extra and point it at your own data:

```bash
uv tool install "prompt-analytics-for-claude-code[dashboard]"
prompt-analytics extract       # write the CSVs the dashboard reads (./output)
prompt-analytics dashboard     # launches Streamlit on http://localhost:8501
```

No data yet, or just curious? The [**live demo**](https://prompt-analytics-demo.streamlit.app) runs the same dashboard on synthetic data. Install, data-dir resolution, and a tour of every page are in the wiki ([Dashboard](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Dashboard)).

## What it can tell you

Per-prompt rows answer questions a daily total can't. Every block below is real output from the bundled demo dataset (`--from-csv demo_data`); on your machine the source line reads `live parse of ~/.claude/projects`.

### What one prompt actually costs, by session depth

The view report-style tools don't give: cost **per prompt**, bucketed by its position in the session. Here the **median prompt at depth 21–50 carries ×2.71 the depth-1 context (≈344k tokens)** — paid as cache rent on *every* turn until `/compact` resets it. `sessions --depth` prints the full distribution; the dashboard draws it as a box plot.

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
  Over the 175-day window your usage is worth $431.85 of anthropic API
  ($74.03/month projected at this rate).
  At this rate the Claude Pro plan ($20.00/mo) pays for itself: your
  API-equivalent is $74.03/month (x3.70 the price), an estimated $54.03/month
  cheaper than paying per token.
```

## Commands

Every analysis command works **on the fly** — it parses your JSONL in memory (~0.5 s) when there is no fresh `extract` to read, so you never have to run `extract` first. All of them accept `--output-dir DIR`, `--format table|csv|json`, `--pricing PATH`, `--no-cache`, and `--from-csv DIR` (analyze the CSVs in DIR as-is, with no live parse and no freshness check — the CLI counterpart of what the dashboard does, e.g. `summary --from-csv demo_data`).

**Totals and breakdowns**

| Command | Key flags | What it shows |
| --- | --- | --- |
| `summary` | | Sessions, prompts, tokens by type, cost per provider, period, subagent share. |
| `by-project` | `--pareto` `--provider` | Cost / tokens / prompts per project, sorted; `--pareto` adds share + cumulative %. |
| `by-model` | `--compact` `--provider` | Token and cost split per model (cache writes split by TTL, subagent column); `--compact` fits 80 columns. |
| `by-token-type` | `--provider` | Cost split per token type — the **context-rent** share of the bill (cache vs generation vs input). |
| `by-category` | `--provider` | Cost and observed complexity per category (needs `categorize`). |
| `prompts` | `--top N` `--provider` | The N most expensive prompts, with a preview. |
| `sessions` | `--depth` \| `--top N`, `--project NAME` | Sessions ranked by cost, or `--depth` for the marginal-cost-by-depth meta-analysis; `--project` restricts to one. |

Beyond these, there are **power-user analyses** on the request grain (`context`, `ttl`, `compactions`, `overhead`, `model-category`, `recommend`, `burn-rate`, `break-even`) and the **pricing / export / pipeline** commands (`compare`, `export`, `extract`, `snapshot`, `categorize`, `run`, `dashboard`, `config init`). The full reference, grouped by purpose with every flag, is in the wiki:

- 📖 [**CLI Commands** — power-user analyses](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/CLI-Commands#power-user-analyses-request-grain)
- 📖 [**CLI Commands** — pricing, export & pipeline](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/CLI-Commands#pricing-export-and-pipeline)

Run `prompt-analytics <command> --help` for the full flag list. `extract` and `run` always regenerate the whole output atomically — there is no incremental mode to misuse, and changing the pricing never requires a re-extract (costs are computed at read time from raw counts).

## Exporting data

`extract` writes the canonical, inspectable, zero-dependency format: relational CSVs keyed by `prompt_id` / `session_id` (`sessions`, `prompts`, `prompts_text`, `tokens`, `token_types`, plus `quota_log` from `snapshot`) into `./output` (gitignored). `tokens.csv` holds **raw counts only** — prices are computed at read time, so a pricing change never needs a re-extract. The exact column layout is the single source of truth in [`prompt_analytics/schema.py`](prompt_analytics/schema.py); the full data flow and CSV contract are in the wiki ([Architecture → the data model](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/wiki/Architecture#the-data-model-csv-contract)).

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

A weighted FR+EN regex classifier labels each prompt across eleven categories (plan / implementation / debug / refactor / review / test / docs / ops / question / followup / other), and an **observed** complexity 1–5 is derived from the effort it actually triggered (turns, tool calls, length, cost) — a measurement, not a guess. Read `$/prompt (med)` alongside the total: a category can top the bill on volume alone, but a high median per prompt flags work that is *intrinsically* expensive (here `debug` is costly both ways at $0.73/prompt, while `followup` stays cheap at $0.26).

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

`Token %` and `Cost %` sit next to their own columns throughout, so volume and spend are never conflated.

</details>

<details>
<summary><strong>Optional LLM refinement</strong></summary>

`categorize --llm` re-labels prompts with an LLM; it only overwrites heuristic rows, never the reverse. Providers:

- `--provider anthropic` (default when `ANTHROPIC_API_KEY` is set; `--batch` uses the Message Batches API, −50% cost).
- `--provider openrouter` — **a third party.** Up to ~2000 characters of each prompt are sent to OpenRouter; the command warns you at runtime before sending.
- `--provider ollama` — a **local** model via the OpenAI-compatible API at `localhost:11434/v1`. Free, no key, nothing leaves your machine.

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
