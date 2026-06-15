# prompt-analytics-for-claude-code

> **Unofficial — not affiliated with Anthropic.** This is a community tool that reads Claude Code's local log files. "Claude" and "Claude Code" are trademarks of Anthropic.

[![CI](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/actions/workflows/ci.yml/badge.svg)](https://github.com/romainfjgaspard/prompt-analytics-for-claude-code/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/prompt-analytics-for-claude-code.svg)](https://pypi.org/project/prompt-analytics-for-claude-code/)
[![Python](https://img.shields.io/pypi/pyversions/prompt-analytics-for-claude-code.svg)](https://pypi.org/project/prompt-analytics-for-claude-code/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- [![Live demo](https://img.shields.io/badge/live-demo-brightgreen.svg)](https://prompt-analytics-demo.streamlit.app) — enable once the demo is deployed (phase 12) -->

# Prompt-level analytics for Claude Code

Other tools tell you what you spent **per day** or **per session**. This one goes down to the **prompt**: every prompt you have ever sent becomes one row, priced from the raw token counts in your local `~/.claude/projects/**/*.jsonl` logs — no account, no API key, nothing leaves your machine.

- **Tokens and cost per prompt and per project.** A real dataset, not a daily total: which prompts and which projects actually cost you money, with a Pareto view of where it concentrates.
- **Meta-analyses across long sessions.** How the marginal cost of a prompt changes with its depth in the session, how the cache read/write mix shifts — the questions you can only ask once you have per-prompt rows.
- **Automatic categorization of your prompts.** A local, zero-dependency heuristic labels every prompt across eleven categories (plan / implementation / debug / refactor / review / test / docs / ops / question / followup / other) and scores its observed complexity 1–5. An optional LLM pass refines it (opt-in — see [Categorization](#categorization)).

It validates its token totals against [ccusage](https://github.com/ryoppippi/ccusage) (see [How the numbers stay accurate](#how-the-numbers-stay-accurate)). Think of it as the per-prompt dataset layer that sits alongside the report-style tools.

## See it without installing anything

**Two surfaces on the same data — a terminal CLI and a Streamlit dashboard.**

Every CLI command is a single line over your local logs. For example, `by-category` auto-labels each prompt and scores its *observed* complexity 1–5 — a local heuristic, no LLM required:

![`prompt-analytics by-category` in the terminal](docs/screenshots/cli-by-category.png)

And a live Streamlit demo runs on a synthetic dataset (no real prompts), so you can explore the dashboard before installing — **the image below links to the live demo:**

**▶ Live demo:** _coming soon_ <!-- https://prompt-analytics-demo.streamlit.app — deploy in phase 12, then enable the badge above AND confirm this URL matches the deployed app -->

[![The prompt-analytics Streamlit dashboard — KPI cards, cost by token type, and daily cost by model](docs/screenshots/dashboard-home.png)](https://prompt-analytics-demo.streamlit.app)

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

Per-prompt rows turn into answers a daily total can't give. Every block below is real output from the bundled demo dataset (`--from-csv demo_data`); on your machine the source line reads `live parse of ~/.claude/projects`.

### Where the bill actually goes: context rent

Almost none of your spend is the text you type or the text the model writes. It is **context rent** — the cache reads and writes that re-send the conversation on every turn. `by-token-type` puts a number on it:

```text
Cost by token type (anthropic)
+-------------------------------------------------------------------------------+
| Token type                         |      Tokens | Token % |    Cost | Cost % |
|------------------------------------+-------------+---------+---------+--------|
| Cache read                         | 497,973,616 |   94.0% | $198.81 |  46.0% |
| Cache write (5m)                   |  22,683,667 |    4.3% | $120.30 |  27.9% |
| Output                             |   2,961,653 |    0.6% |  $61.02 |  14.1% |
| Cache write (1h)                   |   4,680,558 |    0.9% |  $40.97 |   9.5% |
| Input                              |   1,729,726 |    0.3% |   $8.11 |   1.9% |
| Server tool use (req, billed sep.) |         264 |       - |   $2.64 |   0.6% |
| TOTAL                              | 530,029,220 |  100.0% | $431.85 | 100.0% |
+-------------------------------------------------------------------------------+
  Context rent (cache reads + writes): 83.4% of the bill;
  generation (output): 14.1%; fresh input: 1.9%.
```

And it **grows with session depth**: `context` shows the context one turn re-reads by position in the session — on this dataset the median prompt at depth 21–50 carries ×2.71 the depth-1 context (≈344k tokens), and every turn pays that rent until `/compact` or a new session resets it.

### TTL expiry and compaction: the hidden cache costs

Cache entries expire (5-minute or 1-hour TTL); pause longer and the next turn pays to **re-write** the whole context. `ttl` attributes those re-writes to the pause that caused them — here ≈$36 of re-writes across pauses of 5m–1h, plus a little more after longer ones.

`compactions` does the mirror analysis for `/compact`: context before/after and the cache-rebuild cost of the first post-compaction turn (on this dataset, 3 events, a median 266,598 → 85,905 tokens, −67.8%, rebuilt for $0.43 total). `overhead` isolates the fixed per-session cost (system prompt + CLAUDE.md + MCP tools ≈ 118,949 tokens here, $0.69 median per session).

### Subagents, billed back to the prompt that spawned them

Sub-agent (`isSidechain`) work is parsed and its cost rolled into the parent prompt, so nothing is under-counted. `summary` calls it out (`Subagents: $6.19 (1.4% of anthropic cost)`) and `by-model` adds a per-model Subagents column.

### Is a Pro/Max plan worth it vs the API?

`break-even` prices your whole history on the per-token grid (the API-equivalent), projects it to a month, and compares it to each flat-rate plan:

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

With a `quota_log.csv` (written by `snapshot`), the notes also report the peak plan utilization seen in each window. Other "what to do" commands: `model-category --whatif MODEL` re-prices every prompt on another model, `recommend` estimates what compacting long sessions earlier would have saved, and `burn-rate` tracks $/day week over week.

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

<details>
<summary><strong>More commands</strong> — power-user analyses, pricing, export &amp; pipeline</summary>

**Power-user analyses** (request grain — see [What it can tell you](#what-it-can-tell-you))

| Command | Key flags | What it shows |
| --- | --- | --- |
| `context` | `--provider` | Accumulated context per turn by session depth — the "time to `/compact`" signal. |
| `ttl` | `--provider` | Cache-TTL expiry losses: what inter-prompt pauses cost in cache re-writes. |
| `compactions` | `--provider` | Each `/compact` event: context before/after and the cache-rebuild cost. |
| `overhead` | `--provider` | Fixed per-session overhead (system prompt + CLAUDE.md + MCP tools). |
| `model-category` | `--whatif MODEL` `--provider` | Cost by model × category; `--whatif` re-prices every cell on another model. |
| `recommend` | `--min-prompts N` `--compact-at K` | Prescriptive: what compacting long sessions earlier would have saved. |
| `burn-rate` | `--weeks N` `--provider` | Spend trend: $/day and week over week. |
| `break-even` | `--provider` | Plan break-even: your API-equivalent value vs a Pro/Max subscription. |

**Pricing, export and pipeline**

| Command | Key flags | What it shows |
| --- | --- | --- |
| `compare` | `--providers A,B` | The same usage priced on several grids side by side. |
| `export` | `--flat` `--out PATH` | Denormalized export for Excel/BI (see below). |
| `extract` | `--no-text` `--since` `--until` `--timezone` `--strict` | Write the normalized CSVs to `--output-dir`. |
| `snapshot` | | Append current quota utilization to `quota_log.csv`. |
| `categorize` | `--llm` `--provider` `--batch` `--model` `--limit` | Label prompts — heuristic by default, LLM opt-in. |
| `run` | `--categorize` (+ `--llm` `--provider` `--batch` `--model`) `--no-text` `--since` … | Pipeline: extract (+ optional categorize, with full LLM passthrough) + snapshot. |
| `dashboard` | `--output-dir` | Launch the Streamlit dashboard — **reads an extract**, so run `extract` first (needs the `dashboard` extra). |
| `config init` | | Write a default `config.yml` into the output directory. |

</details>

Run `prompt-analytics <command> --help` for the full flag list. `extract` and `run` always regenerate the whole output atomically — there is no incremental mode to misuse, and changing the pricing never requires a re-extract (costs are computed at read time from raw counts).

## Exporting data

`extract` writes the canonical, inspectable, zero-dependency format: relational CSVs keyed by `prompt_id` / `session_id` (`sessions`, `prompts`, `prompts_text`, `tokens`, `token_types`, plus `quota_log` from `snapshot`) into `./output` (gitignored). `tokens.csv` holds **raw counts only** — prices are computed at read time, so a pricing change never needs a re-extract. The exact column layout is the single source of truth in [`prompt_analytics/schema.py`](prompt_analytics/schema.py); the full data flow is documented in [`docs/architecture.md`](docs/architecture.md).

For spreadsheets and BI tools that want one flat table, `export --flat` writes a single denormalized CSV — one row per prompt, token columns pivoted (`input_tokens`, `output_tokens`, `cache_read_tokens`, …), a `cost_<provider>_usd` column per provider, and the session fields duplicated in:

```bash
prompt-analytics export --flat --out my_usage.csv
```

Every command also takes `--format json` (a `{title, rows, notes}` object with raw numeric values) and `--format csv` (raw rows on stdout, notes on stderr) for piping into `jq`, pandas, or a notebook.

## Pricing providers

Costs are computed at read time from a multi-provider pricing grid, [`prompt_analytics/data/pricing.yml`](prompt_analytics/data/pricing.yml). Two providers ship by default:

- **`anthropic`** — Anthropic's published API rates (Opus / Sonnet / Haiku / Fable, with `cache_read` at 0.1×, `cache_write_5m` at 1.25×, `cache_write_1h` at 2× of base input).
- **`copilot`** — the GitHub Copilot equivalent, so you can answer *"what would this usage cost billed through Copilot instead of the Anthropic API?"*

That comparison is exactly what `compare` is for:

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

The grid is **generic** — add a key under `providers:` to price the same usage on *any* rate card (an internal Copilot plan, an AWS Bedrock tier, a negotiated rate), then `compare --providers anthropic,my-company`, or pass your own with `--pricing ./my-pricing.yml`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the grid schema and lookup rules. The bundled grid is kept honest by a weekly CI drift job that diffs it against LiteLLM and the live Copilot pricing page.

## Categorization

By default, categorization is **100% local and runs for everyone** — no API key, no network:

```bash
prompt-analytics categorize        # heuristic, writes categories.csv
prompt-analytics by-category
```

- **Category** — a weighted FR+EN regex classifier (`classifier_model = "heuristic-v2"`) labels each prompt one of eleven categories: `plan` / `implementation` / `debug` / `refactor` / `review` / `test` / `docs` / `ops` / `question` / `followup` / `other`. The French patterns tolerate missing accents, and the agentic categories (`review`, `test`, `docs`, `ops`) catch the long multi-step prompts the first six missed — on real history "other" drops from 60.8% to 17.6% of prompts. A heuristic re-run upgrades rows stamped by an older heuristic version; LLM-classified rows are never overwritten.
- **Complexity 1–5** — *observed*, not guessed: derived from the effort the prompt actually triggered (quantiles over `assistant_turns`, `tool_calls`, `char_count`, and computed cost). A measurement, not an estimate.

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

The `$/prompt (med)` column is the one to read alongside the total: a category can top the bill just by volume, but a high median per prompt is what flags work that is *intrinsically* expensive — here `debug` is costly both ways ($0.73/prompt) while `followup` stays cheap ($0.26). `Token %` and `Cost %` sit next to their own columns throughout, so volume and spend are never conflated.

**LLM refinement (opt-in).** `categorize --llm` re-labels prompts with an LLM; it only overwrites heuristic rows, never the reverse. Providers:

- `--provider anthropic` (default when `ANTHROPIC_API_KEY` is set; `--batch` uses the Message Batches API, −50% cost).
- `--provider openrouter` — **a third party.** Up to ~2000 characters of each prompt are sent to OpenRouter; the command warns you at runtime before sending.
- `--provider ollama` — a **local** model via the OpenAI-compatible API at `localhost:11434/v1`. Free, no key, nothing leaves your machine.

Set keys in a `.env` file (see [`.env.example`](.env.example)). Most Claude Code Pro/Max subscribers have no Console API key — that is exactly why the heuristic is the default.

## Quota snapshot

`snapshot` records your current plan utilization into `quota_log.csv` so you can chart it over time (the real-time tools show it but don't keep history).

> **Disclaimer.** `snapshot` reads an **undocumented** OAuth endpoint (`https://api.anthropic.com/api/oauth/usage`), reusing the OAuth token Claude Code already stored in `~/.claude/.credentials.json`. That token grants access to your account; this command sends it to Anthropic's own endpoint and nothing else. The endpoint is not part of any public API and may change or break without notice — the command fails gracefully if it does. The endpoint was discovered by [usage-monitor-for-claude](https://github.com/jens-duttke/usage-monitor-for-claude); we send an honest `prompt-analytics-for-claude-code/<version>` User-Agent (no spoofing).

## How the numbers stay accurate

Claude Code writes a single assistant message as **several JSONL lines carrying progressive snapshots** of the same response — `output_tokens` grows line by line, and the model can even change mid-message. Naively summing every line double-counts and inflates token totals by roughly 2.5×.

The parser counts each response exactly once: for a given `message.id + requestId` it keeps the **largest** usage snapshot (ties broken by the first line, so a message straddling midnight belongs to its start day). Deduplication is **global across all files**, not per session — that is what corrects resumed / `--resume` sessions, where the same records are replayed into a new file.

On top of dedup, the extractor also:

- **Filters fake prompts** — `isMeta` entries, `<command-name>` / `<local-command-stdout>` blocks, `[Request interrupted by user]`, and the synthetic post-compaction "continuation" message are kept out of `prompts.csv` and out of categorization, while their token usage is still counted against the session.
- **Has an explicit sidechain / sub-agent policy** — inline `isSidechain` events and separate `subagents/*.jsonl` files are parsed and their cost is rolled into the parent prompt (so nothing is silently under-counted), but they are excluded from the parent's `assistant_turns` / `tool_calls`.
- **Splits cache writes by TTL** — `cache_write_5m` (1.25×) vs `cache_write_1h` (2×), and counts `server_tool_use` requests separately (billed per request, not per token).

These totals reconcile **bucket-for-bucket with `bunx ccusage --json`** (day × model) on real history — the reconciliation is scripted in `scripts/reconcile_ccusage.py`. Each command also prints its data source, and `extract` ends with a loud report (files read / skipped, unknown event types, unpriced models, Claude Code versions seen) so a silent format break surfaces immediately.

## Privacy

This tool reads your entire Claude Code prompt history, so it is explicit about what it touches.

**Reads (locally):** `~/.claude/projects/**/*.jsonl` (read-only); `~/.claude/.credentials.json` (only `snapshot`, to reuse your existing token); the pricing grid bundled in the package. A relocated Claude directory is honored via `CLAUDE_CONFIG_DIR`, like Claude Code itself. On Windows, `~` means `C:\Users\<you>` (`%USERPROFILE%`).

**Writes (locally, never inside the installed package):** the CSVs in your output directory (default `./output`, gitignored). `prompts_text.csv` contains the **full text of every prompt in clear**, and `prompts.csv` keeps a short `prompt_preview` — treat an export as sensitive before sharing or committing it. A parse cache lives under `%LOCALAPPDATA%\prompt-analytics\` on Windows, `$XDG_CACHE_HOME/prompt-analytics/` (default `~/.cache/prompt-analytics/`) elsewhere — safe to delete; entries for deleted logs are garbage-collected on each run.

- `--no-text` is honest: it skips `prompts_text.csv` **and** blanks the `prompt_preview` column.
- Free-text cells starting with `=`, `+`, `-`, or `@` are prefixed with `'` so a crafted prompt can't execute as a spreadsheet formula.

**Network:** **nothing by default.** `extract` and every analysis command are fully local. `snapshot` calls Anthropic's OAuth usage endpoint with your token (see disclaimer). `categorize --llm` sends prompt excerpts to the provider you choose — and OpenRouter is a third party.

## Related tools

- **[ccusage](https://github.com/ryoppippi/ccusage)** — fast terminal usage reports (daily/session/blocks) and JSON output, 14+ CLIs supported. Complementary: it does aggregated reports, this does the per-prompt dataset. Token totals here are validated against it.
- **[usage-monitor-for-claude](https://github.com/jens-duttke/usage-monitor-for-claude)** — real-time quota tray app for Windows; source of the quota-endpoint discovery credited above.
- **[Claude-Code-Usage-Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)** — real-time 5-hour-block burn-rate monitor with P90 predictions.
- **Claude Code Analytics API** — official, org-level analytics; requires an organization and an Admin API key.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, how to add a pricing provider, and how to capture a versioned test fixture. In short: `uv sync --all-extras`, then `ruff check .`, `ruff format --check .`, `mypy prompt_analytics tests`, and `pytest`.

## License

[MIT](LICENSE)
