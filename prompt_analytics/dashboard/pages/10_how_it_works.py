"""How it works: the trust argument, displayed — not folded in expanders (7.8).

The deduplication story is the project's best credibility asset; this page
shows it (and the validation against ccusage) directly, with the secondary
topics in half-width columns below. The closing section makes the **cost
inputs** transparent: the per-token grid the dashboard prices on, the Claude
flat-rate plans, and the GitHub Copilot credit allowances (the three sources
every dollar on the board is derived from).

No charts here: this is the only analytics page rendered headless under
``AppTest`` (``tests/test_dashboard.py``), so it stays on native Streamlit
widgets (which follow the light/dark toggle on their own) — never ECharts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from prompt_analytics import pricing
from prompt_analytics.dashboard import data, filters, theme

# Token-type columns shown in the per-token rate grid, in billing order.
_RATE_COLS = {
    "input": "Input",
    "cache_read": "Cache read",
    "cache_write_5m": "Cache write (5m)",
    "cache_write_1h": "Cache write (1h)",
    "output": "Output",
}


def _rate_grid(provider: str) -> pd.DataFrame:
    """The per-token rate grid (USD / 1M tokens) for ``provider``, one row per model."""
    grid = pricing.load_pricing()["providers"].get(provider, {})
    models: dict[str, Any] = grid.get("models", {})
    rows = [
        {"Model": model, **{label: float(rates[key]) for key, label in _RATE_COLS.items()}}
        for model, rates in models.items()
    ]
    return pd.DataFrame(rows, columns=["Model", *_RATE_COLS.values()])


def _plans_df() -> pd.DataFrame:
    """Claude flat-rate subscription plans (the break-even axis)."""
    plans = pricing.load_plans()
    rows = [
        {"Plan": p.get("label", name), "Subscription ($/mo)": float(p["monthly_usd"])}
        for name, p in plans.items()
    ]
    return pd.DataFrame(rows, columns=["Plan", "Subscription ($/mo)"])


def _copilot_df() -> pd.DataFrame:
    """GitHub Copilot tiers: subscription + bundled AI-credit allowance."""
    plans = pricing.load_copilot_plans()
    rows = [
        {
            "Tier": p.get("label", name),
            "Subscription ($/mo)": float(p["monthly_usd"]),
            "Included usage ($/mo)": float(p["included_usd"]),
            "Included credits": int(round(float(p["included_usd"]) / 0.01)),
        }
        for name, p in plans.items()
    ]
    return pd.DataFrame(
        rows,
        columns=["Tier", "Subscription ($/mo)", "Included usage ($/mo)", "Included credits"],
    )


def _source_md(url: str | None, label: str) -> str:
    """A trailing markdown link to a price source (empty when none is declared)."""
    return f" · source: [{label}]({url})" if url else ""


def _render_cost_inputs() -> None:
    """Make every cost on the board traceable to its source (product TODO)."""
    primary = data.primary_provider()
    pricing_data = pricing.load_pricing()
    updated = pricing_data.get("updated_at", "—")
    primary_source = pricing_data.get("providers", {}).get(primary, {}).get("source")

    theme.section(
        "The costs behind every number",
        "Every dollar on this dashboard is computed at read time from the grid "
        "below — nothing is baked into the data. Edit `pricing.yml` to match your "
        "own contract and the whole board re-prices.",
    )

    st.subheader("Per-token rates")
    st.caption(
        theme.md_escape(
            f"USD per 1,000,000 tokens, for the primary provider (`{primary}`). "
            "Token-type semantics: cache reads are the cheap tier (~0.1x input), "
            "5-minute cache writes ~1.25x and 1-hour writes ~2x input, output the "
            f"dearest. Rates from `pricing.yml` (updated {updated})."
        )
        + _source_md(primary_source, f"{primary} pricing")
    )
    grid = _rate_grid(primary)
    money: dict[Any, str | Callable[[object], str] | None] = {
        col: "${:,.2f}" for col in _RATE_COLS.values()
    }
    st.dataframe(
        grid.style.format(money, na_rep="—"),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        theme.md_escape(
            "Server-side tools (e.g. web search) are billed per request, not per "
            "token (Anthropic: $0.01/search), and are added on top of the grid above."
        )
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Claude subscription")
        st.caption(
            "Flat-rate plans the **Quotas → break-even** compares against your "
            "API-equivalent usage. List prices in USD/month."
            + _source_md(pricing_data.get("plans_source"), "Anthropic pricing")
        )
        plans = _plans_df()
        if plans.empty:
            st.info("No `plans:` section in pricing.yml.")
        else:
            st.dataframe(
                plans.style.format({"Subscription ($/mo)": "${:,.2f}"}),
                width="stretch",
                hide_index=True,
            )
    with right:
        st.subheader("GitHub Copilot (credits)")
        st.caption(
            theme.md_escape(
                "Copilot is usage-based since 2026-06-01: 1 GitHub AI Credit = "
                "$0.01, and each tier bundles a monthly credit allowance. The "
                "**Quotas → Cost via GitHub Copilot** section prices your usage "
                "as subscription + overage beyond the allowance."
            )
            + _source_md(pricing_data.get("copilot_plans_source"), "GitHub billing docs")
        )
        copilot = _copilot_df()
        if copilot.empty:
            st.info("No `copilot_plans:` section in pricing.yml.")
        else:
            st.dataframe(
                copilot.style.format(
                    {
                        "Subscription ($/mo)": "${:,.2f}",
                        "Included usage ($/mo)": "${:,.2f}",
                        "Included credits": "{:,}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


def main() -> None:
    """Render the How it works page (static content, no data required)."""
    st.title("How it works")
    # This page has no sidebar filters; keep any selection from other pages alive.
    filters.persist_filters()

    st.markdown(
        "`~/.claude/projects/*.jsonl` → **`extract`** (offline) → CSVs → "
        "**analytics** → CLI tables & this dashboard."
    )

    theme.section("Your data stays on your machine")
    st.markdown(
        "Claude Code writes a JSONL file per session under `~/.claude/projects/`. "
        "Each file records every user prompt and assistant response together "
        "with token usage and metadata. The tool reads these files directly "
        "from your machine: the `extract` command is **100% offline** and never "
        "contacts a remote service. Prompt text stays out of the CSVs unless "
        "you opt in (`prompt_text` in `config.yml`)."
    )

    theme.section("Why you can trust the totals")
    left, right = st.columns(2)
    with left:
        st.subheader("Deduplication")
        st.markdown(
            "A single assistant message is written to the JSONL as **several "
            "lines** carrying *progressive* usage snapshots: as the message "
            "grows, `output_tokens` increases and the model can even change "
            "mid-message. Naively summing usage across those lines would "
            "multi-count the same API call.\n\n"
            "The extractor keys every usage record by `message.id` + "
            "`requestId` and keeps the **largest snapshot per key**. The "
            "deduplication is **global across all files**, not per session: "
            "resumed, forked and compacted sessions replay the same records "
            "into new files with a new `sessionId`, and only the global key "
            "catches those replays."
        )
    with right:
        st.subheader("Validation")
        st.markdown(
            "- Totals are **reconciled against ccusage** (the community "
            "reference) on real history: day × model buckets must match "
            "exactly (`scripts/reconcile_ccusage.py`).\n"
            "- The request grain (`requests.csv`, one row per deduplicated API "
            "request) **sums back to the per-prompt totals to the token** — "
            "an invariant every schema change must re-prove.\n"
            "- Costs are computed **at read time** from the `pricing.yml` "
            "grid — editable, per provider — never baked into the data; "
            "server-tool requests (e.g. web search) are billed per request.\n"
            "- The dashboard KPIs match `prompt-analytics summary`: same "
            "dedup, same pricing grid."
        )

    theme.section("What the analyses add")
    left, right = st.columns(2)
    with left:
        st.subheader("Categorization")
        st.markdown(
            "`categorize` labels each prompt with a work category (debug, "
            "implementation, ops, review, …) and a 1-5 complexity score. The "
            "default classifier is a **local, stdlib-only heuristic** "
            "(weighted FR+EN keyword scoring) that needs no API key and no "
            "network. The complexity score is *observed* — derived from the "
            "effort a prompt actually triggered (assistant turns, tool calls, "
            "characters, cost), not guessed.\n\n"
            "An optional `categorize --llm` pass refines the labels with an "
            "LLM (Anthropic, OpenRouter, or local Ollama). It only overwrites "
            "heuristic rows, never the other way around."
        )
    with right:
        st.subheader("Quota snapshots")
        st.markdown(
            "The `snapshot` command calls an undocumented Claude OAuth usage "
            "endpoint (discovered via usage-monitor-for-claude). It logs "
            "utilization percentages and reset times for each window "
            "(five_hour, seven_day, seven_day_sonnet). Because the endpoint "
            "is unofficial it may break without notice, so any failure is "
            "non-blocking and simply skips that snapshot."
        )
        st.subheader("Request grain")
        st.markdown(
            "`requests.csv` keeps one row per API request (timestamp, model, "
            "pivoted token counts, sidechain and post-compaction flags). It "
            "is what powers the TTL, compaction and accumulated-context "
            "analyses on the **Optimize** page and in the CLI."
        )

    _render_cost_inputs()


main()
