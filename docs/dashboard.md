# Dashboard

The dashboard is an interactive Streamlit view over your extracted data: cost
breakdowns, the session-depth meta-analysis, per-model and per-category charts,
and quota trends. Charts are **Apache ECharts** (via `streamlit-echarts`) on a
dark-by-default theme with a native light/dark toggle, and most charts
**cross-filter the whole board on click** (and the time trend on a date brush).
It is an **optional extra** (it pulls in Streamlit, Apache ECharts, pandas and
NumPy), so it is not installed by the core tool.

## Install & launch

With an editable checkout (uv):

```bash
uv sync --extra dashboard
prompt-analytics dashboard            # reads ./output by default
```

Without installing, straight from PyPI (once published):

```bash
uvx --with "prompt-analytics-for-claude-code[dashboard]" prompt-analytics dashboard
```

Or with pip:

```bash
pip install "prompt-analytics-for-claude-code[dashboard]"
prompt-analytics dashboard
```

(Double quotes work in every shell; single quotes are passed through literally
by cmd.exe and break the install.)

The `dashboard` command launches `streamlit run` on the bundled app. Streamlit
serves on **http://localhost:8501** by default; if the port is taken it picks
the next free one and prints the URL. Pass Streamlit flags through the
environment or a `.streamlit/config.toml` (for example to change the port or
theme) — see the [Streamlit docs](https://docs.streamlit.io/).

The dashboard renders **already-extracted CSVs**. Run `prompt-analytics extract`
(or `run`) first; if you start an `extract` while the dashboard is open, the
cache keys on the CSV mtimes, so a browser refresh shows the new data.

## Choosing the data directory

Resolution order:

1. `CCA_DEMO=1` — use the bundled synthetic `demo_data/` dataset and show a
   "Demo data" banner. This is what the public live demo runs.
2. `CCA_DATA_DIR` — an explicit directory of CSVs.
3. `--output-dir DIR` on the `dashboard` command (exported to the app as
   `PROMPT_ANALYTICS_OUTPUT_DIR`).
4. Default: `./output`.

```bash
# Linux / macOS
CCA_DEMO=1 prompt-analytics dashboard           # synthetic demo data
CCA_DATA_DIR=/path/to/csvs prompt-analytics dashboard
prompt-analytics dashboard --output-dir ./out   # a specific extract
```

```powershell
# Windows (PowerShell) -- the `VAR=x cmd` prefix syntax does not exist there
$env:CCA_DEMO = "1"; prompt-analytics dashboard
$env:CCA_DATA_DIR = "C:\path\to\csvs"; prompt-analytics dashboard
prompt-analytics dashboard --output-dir .\out   # a specific extract

# Unset a variable when you are done:
Remove-Item Env:CCA_DEMO
```

## Pages

All charts share one visual identity: color *semantics* live in
`dashboard/theme.py` (stable colors per token type, per model family — opus =
purples, sonnet = blues, haiku = greens, fable = warm — and per category) and
the ECharts render layer in `dashboard/echarts.py` (a theme-aware base so every
chart follows the light/dark toggle on its own). Pages price everything on the
**primary provider** (the first in `pricing.yml`, anthropic by default); the
multi-provider grid is still in `pricing.yml` and the CLI.

Most categorical charts **emit a cross-filter on click** (project / model /
category) and the time trend filters a **date range by brush**; a click on a
chart that is a filter dimension surfaces an **"Explore →"** button into the
Explorer page on the current selection.

| Page | Shows |
| --- | --- |
| Home | **Hero numbers** (context-rent share of the bill, median follow-up cost multiplier) with the cost-split bar, spend-trend KPIs with **sparklines and week-over-week deltas**, and cost over time stacked by model (click a model → filter) with a Day/Week/Month toggle. |
| Usage | Cost by token type per day/week (**date brush + click** → date filter), cache-read headline + non-cache volume bars, **subagent cost** (KPI + a bar per token type), and the **spend punchcard** (weekday × hour). |
| Models | Two donuts (token volume / cost by model), **per-prompt cost distribution by model** (clipped box plot), and cost by model over time (Day/Week/Month toggle). Every chart emits the model filter on click. |
| Prompts | Twin bars (count + cost per category, with median $/prompt), a complexity histogram, and two clipped box plots (cost by category / by complexity). Clicking a category filters the board (needs `categorize`). |
| Session depth | KPI **deep-vs-opener cost multiplier**, a clipped box plot of one prompt's cost by depth band, and **2×2 small-multiples** of the input-side token mix (median + p25–p75 band) — the differentiating analysis. |
| Sessions | **Horizontal cost pareto by project** (click a bar → project filter), a cost treemap (click a tile → Explorer), prompts/session, and a clipped box plot of cost/session by model (click → model filter). |
| Explorer | The detail page: drill **day → session → prompt** (`st.dataframe` tables) and a cumulative-cost timeline; respects the global cross-filters, so it closes the filter-driven block. Reached from the **"Explore →"** button and from treemap tiles. |
| Optimize | The prescriptive story on the request grain: an **avoidable-$ headline** and three cards (the leak = long-pause cache rewrites with TTL bars, the lever = compact earlier with adjustable thresholds, the reassurance = compaction is cheap), detail tables in expanders. |
| Quotas | **Quota windows first** — utilization gauges with green/amber/red zones, the trend with **reset windows overlaid**, reset countdowns (need `snapshot`) — then the cost story: **plan break-even** (your API-equivalent vs each subscription) and **Cost via GitHub Copilot** (AI credits: subscription + overage beyond the allowance). |
| How it works | The trust argument displayed: deduplication, ccusage reconciliation, pricing-at-read-time, categorization, quota snapshots — and **the cost inputs** (the per-token rate grid, Claude flat-rate plans, and GitHub Copilot credit allowances). |

## Configuration (`config.yml`)

The dashboard reads `config.yml` from the active data directory. Create the
default with `prompt-analytics config init`. It currently has one section:

```yaml
features:
  categorization: true   # gate the Prompts page
  prompt_text: true      # whether extract stored full prompt text
  quota_snapshot: true   # gate the Quotas page
```

Missing keys fall back to these defaults. A `features` flag set to `false`
hides the corresponding page; it does not delete any data.
