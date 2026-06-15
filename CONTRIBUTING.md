# Contributing

Thanks for your interest in `prompt-analytics-for-claude-code`. This is a small,
quasi-stdlib tool with a strict quality bar (typed, tested, linted). PRs are
welcome — please run the checks below before submitting.

## Dev setup

The project uses [uv](https://docs.astral.sh/uv/). Clone, then sync all extras
(core + `categorize` + `dashboard` + `dev`):

```bash
uv sync --all-extras
```

Run the full local CI — the same steps the GitHub workflow runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy prompt_analytics tests
uv run pytest                       # coverage gate: 85%
```

Architecture and the data flow (`extract → analytics → {cli, dashboard, csv}`)
are documented in [`docs/architecture.md`](docs/architecture.md). The CSV column
contract lives in one place, [`prompt_analytics/schema.py`](prompt_analytics/schema.py)
— change a column there and every writer/reader follows.

A few ground rules:

- **The core stays light.** Core depends only on `python-dotenv`, `pyyaml`, and
  `rich`. Heavy deps belong in extras: `anthropic`/`openai` in `categorize`,
  `streamlit`/`plotly`/`pandas`/`numpy` in `dashboard`. Don't import an extra
  from core code.
- **`tokens.csv` stores raw counts, never costs.** Costs are computed at read
  time in `analytics.py` from the pricing grid, so a pricing change never
  requires a re-extract. Keep it that way.
- **Every bug fix gets a regression test.** The happy path is not enough.

## Adding a pricing provider

Pricing is a generic multi-provider grid in
[`prompt_analytics/data/pricing.yml`](prompt_analytics/data/pricing.yml). To add
a provider (your company's internal rates, a Bedrock tier, …), add a key under
`providers:`. All prices are **USD per 1,000,000 tokens**:

```yaml
providers:
  my-company:
    models:
      claude-opus-4-8:
        input: 4.50          # negotiated rate
        output: 22.50
        cache_read: 0.45
        cache_write_5m: 5.625
        cache_write_1h: 9.00
    fallbacks:               # matched by longest model-name prefix
      claude-opus:
        input: 4.50
        output: 22.50
        cache_read: 0.45
        cache_write_5m: 5.625
        cache_write_1h: 9.00
```

Then use it with `--providers anthropic,my-company` on `compare`, or `--provider
my-company` on the other commands. Users can keep their grid out of the repo and
pass it with `--pricing ./my-pricing.yml`.

Lookup rules (see `pricing.get_model_pricing`): an exact model id wins; otherwise
the **longest matching prefix** under `fallbacks` is used; `[1m]` and
long-context suffixes are stripped before lookup. An unpriced model is never
silently zeroed — it surfaces in the extraction/analytics report so you know to
add an entry.

The bundled `anthropic` and `copilot` grids are validated weekly by CI
(`.github/workflows/pricing-drift.yml`): the job diffs `anthropic` against
LiteLLM's `model_prices_and_context_window.json` and re-runs
`scripts/fetch_copilot_pricing.py` against the live Copilot pricing page, opening
a PR if either drifts. Update those two grids through that job, not by hand.

## Capturing a test fixture

Claude Code's JSONL format changes without notice, so parsing is pinned against
fixture files **per Claude Code version**, under
`tests/fixtures/claude-code-<version>/`. When you hit a new format, capture one
from your own logs — anonymized, so it is safe to commit:

```bash
uv run python scripts/capture_fixture.py path/to/real/session.jsonl
# --version 2.1.180   override the auto-detected version label
# --project demo-app   generic project folder name in the fixture
```

`capture_fixture.py` rewrites the log **character by character** (letters → `x`,
digits → `0`, punctuation and length preserved) while keeping everything the
parser counts: structure, ids, the `uuid`/`parentUuid` attribution chain,
`message.usage`, `model`, `timestamp`, `version`, and the filtering markers
(`<command-name>`, `[Request interrupted…`). It scrubs your username, paths and
prompt text — verify the result before committing. `test_fixtures_versioned.py`
then acts as a drift canary: it asserts the fixture parses cleanly (no invalid
lines, no unknown event types) and deterministically.

## Submitting

- Branch off `main`, keep commits focused, and describe the *why* in the PR.
- Add or update tests and docs (README / `docs/`) alongside the code.
- Update [`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]`.
- Make sure the four checks above pass locally — CI runs them on Python
  3.10–3.14 across Ubuntu and Windows.
