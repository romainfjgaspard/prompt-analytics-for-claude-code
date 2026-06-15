"""Optimize page: the prescriptive analyses (7.4) — what to do, with numbers.

Reworked for clarity (browser review: the three raw analyses read as a jargon
heavy dump with no takeaway). The page now leads with a headline — how much
cache spend looks *avoidable* and which lever dominates — then tells one story in
three cards: the leak (long pauses re-write expired cache), a lever (a few long
sessions could compact sooner), and the reassurance (compacting is nearly free,
so it's the answer). Detail tables live behind expanders.

Reads the raw :class:`Dataset` (request grain), not the filtered pandas frames,
so the global sidebar filters do not apply — stated on the page. Migrated to
Apache ECharts (``docs/MIGRATION-ECHARTS.md``); no chart emits a cross-filter.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics import analytics
from prompt_analytics.dashboard import data, echarts, filters, theme

# Pause buckets where cache has fully or partly expired (everything past 5m).
_EXPIRED_GAPS = ("5m-1h", "1h-6h", "> 6h")
_LONG_GAPS = ("1h-6h", "> 6h")


def _notes_after_source(result: analytics.TableResult) -> list[str]:
    """The human notes of a result, minus the leading source / pricing lines.

    Dollar signs are escaped: a note with two amounts would otherwise open a
    LaTeX math span in ``st.caption``.
    """
    return [theme.md_escape(note) for note in result.notes[1:] if not note.startswith("Pricing ")]


def _ttl_option(rows: list[dict[str, Any]], primary: str) -> dict[str, Any]:
    """Write cost per pause bucket, the estimated expiry loss overlaid.

    The two bars share a baseline (``barGap: -100%``): the light bar is the
    total cache writes after the pause, the darker bar (drawn on top, always
    <= the light one) is the portion estimated to be expiry loss above the
    incremental baseline. Read-only (the page has no global filter dimension).
    """
    gaps = [str(r["gap"]) for r in rows]
    write_cost = [round(float(r["write_cost_usd"] or 0), 2) for r in rows]
    excess = [round(float(r["excess_cost_usd"] or 0), 2) for r in rows]
    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": f"Cost to rebuild cache, by pause length ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 72, "right": 24, "top": 56, "bottom": 56, "containLabel": True}
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "trigger": "axis",
        "axisPointer": {"type": "shadow"},
        "valueFormatter": echarts.js("function(v){return v==null?'-':'$'+Number(v).toFixed(2);}"),
    }
    xaxis = echarts.category_axis(gaps)
    xaxis["name"] = "Pause before resuming"
    xaxis["nameLocation"] = "middle"
    xaxis["nameGap"] = 32
    xaxis["nameTextStyle"] = {"color": c["muted"]}
    option["xAxis"] = xaxis
    yaxis = echarts.value_axis(money=True, name="Cost (USD)")
    yaxis["nameLocation"] = "middle"
    yaxis["nameGap"] = 56
    option["yAxis"] = yaxis
    option["series"] = [
        {
            "type": "bar",
            "name": "Cache writes after the pause",
            "data": write_cost,
            "barWidth": "45%",
            "itemStyle": {"color": "#C4B5FD", "borderRadius": [4, 4, 0, 0]},
        },
        {
            "type": "bar",
            "name": "Est. expiry loss (avoidable)",
            "data": excess,
            "barGap": "-100%",
            "barWidth": "45%",
            "itemStyle": {"color": "#7C3AED", "borderRadius": [4, 4, 0, 0]},
        },
    ]
    return option


def _beforeafter_option(before: int, after: int) -> dict[str, Any]:
    """Tiny horizontal before -> after bar of median context around a compaction."""
    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["grid"] = {"left": 16, "right": 56, "top": 8, "bottom": 8, "containLabel": True}
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "trigger": "item",
        "valueFormatter": echarts.js("function(v){return Number(v).toLocaleString()+' tokens';}"),
    }
    option["xAxis"] = echarts.value_axis()
    option["xAxis"]["axisLabel"]["show"] = False
    option["xAxis"]["splitLine"] = {"show": False}
    option["yAxis"] = echarts.category_axis(["Before", "After"], inverse=True)
    option["series"] = [
        {
            "type": "bar",
            "data": [
                {"value": before, "itemStyle": {"color": theme._rgba("#7C3AED", 0.5)}},
                {"value": after, "itemStyle": {"color": "#10B981"}},
            ],
            "barWidth": "55%",
            "itemStyle": {"borderRadius": [0, 4, 4, 0]},
            "label": {
                "show": True,
                "position": "right",
                "color": c["text"],
                "formatter": echarts.js(
                    "function(p){return (p.value/1000).toFixed(0)+'k tokens';}"
                ),
            },
        }
    ]
    return option


def _render_headline(ttl: analytics.TableResult, rec: analytics.TableResult) -> None:
    """Top headline: how much cache spend looks avoidable, and the main lever."""
    pause_loss = sum(float(r["excess_cost_usd"] or 0) for r in ttl.rows)
    compact_saving = sum(float(r["saving_usd"] or 0) for r in rec.rows)
    avoidable = pause_loss + compact_saving
    st.subheader(f"≈ ${avoidable:,.0f} of cache spend looks avoidable")
    st.write(
        "Caching is mostly a good deal — but two habits quietly cost extra: "
        "**resuming a session after a long pause** (the cache expired, so it is "
        "rebuilt from scratch) and **letting a few sessions run very long** before "
        "compacting. Here is how much, and what to do about it."
    )
    cols = st.columns(3)
    cols[0].metric("Avoidable, total (est.)", f"${avoidable:,.0f}")
    cols[1].metric("From long pauses", f"${pause_loss:,.0f}")
    cols[2].metric("From compacting sooner", f"${compact_saving:,.0f}")


def _render_pauses(ttl: analytics.TableResult, primary: str) -> None:
    """Card 1 — the leak: long pauses re-write expired cache."""
    theme.section(
        "1 · Long pauses are the expensive part",
        "Cache entries expire after 5 minutes (an hour for the long-lived ones). "
        "Come back after a longer pause and the expired context is rewritten — you "
        "pay to rebuild it from scratch.",
    )
    if not ttl.rows:
        st.info("No pause data yet.")
        return
    long_loss = sum(float(r["excess_cost_usd"] or 0) for r in ttl.rows if r["gap"] in _LONG_GAPS)
    resumptions = sum(int(r["events"]) for r in ttl.rows if r["gap"] in _EXPIRED_GAPS)
    cols = st.columns(2)
    cols[0].metric("Rebuilt after pauses > 1h", f"${long_loss:,.2f}")
    cols[1].metric("Resumptions after a pause > 5m", f"{resumptions:,}")
    echarts.render(_ttl_option(ttl.rows, primary), key="optimize_ttl", height="400px")
    st.caption(
        "👉 Wrap up or run **/compact** before stepping away — and starting a "
        "fresh session can be cheaper than resuming a cold one."
    )
    with st.expander("Detail by pause length"):
        df = data.table_df(ttl)
        st.dataframe(
            df.style.format(
                {
                    "Avg cache write": "{:,.0f}",
                    "Write cost": "${:,.2f}",
                    "Est. expiry loss": "${:,.2f}",
                },
                na_rep="— (baseline)",
            ),
            width="stretch",
            hide_index=True,
        )
        for note in _notes_after_source(ttl):
            st.caption(note)


def _render_recommendations(ds: analytics.Dataset, primary: str) -> None:
    """Card 2 — a lever: a few long sessions could compact sooner."""
    theme.section(
        "2 · A few long sessions could compact sooner",
        "Carrying a big context across a very long session keeps paying to re-read "
        "it. Compacting partway trims that. Tune the thresholds to see the estimate "
        "for your longest sessions.",
    )
    left, right = st.columns(2)
    min_prompts = int(
        left.number_input("Flag sessions longer than (prompts)", min_value=5, value=50, step=5)
    )
    compact_at = int(right.number_input("Compact around prompt", min_value=5, value=30, step=5))
    result = analytics.recommendations(ds, primary, min_prompts=min_prompts, compact_at=compact_at)
    if result.rows:
        total_saving = sum(r["saving_usd"] for r in result.rows)
        rent = sum(r["rent_usd"] for r in result.rows)
        cols = st.columns(3)
        cols[0].metric("Context cost paid", f"${rent:,.2f}")
        cols[1].metric("Est. saving if compacted", f"${total_saving:,.2f}")
        cols[2].metric("Sessions concerned", f"{len(result.rows)}")
        with st.expander("Detail by session"):
            df = data.table_df(result)
            st.dataframe(
                df.style.format(
                    {
                        "Cache rent paid": "${:,.2f}",
                        "Est. if compacted": "${:,.2f}",
                        "Est. saving": "${:,.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            for note in _notes_after_source(result):
                st.caption(note)
    else:
        st.caption("No sessions cross these thresholds — nothing to compact sooner.")


def _render_compactions(ds: analytics.Dataset, primary: str) -> None:
    """Card 3 — the reassurance: compacting is nearly free."""
    theme.section(
        "3 · Compacting is cheap — that's the point",
        "Every /compact (auto or manual) in your history: how much context it "
        "dropped and what rebuilding the cache cost. Compared with the pause losses "
        "above, compacting is almost free — which is why it's the answer.",
    )
    result = analytics.compactions(ds, primary)
    if not result.rows:
        st.info("No compactions found in the history.")
        return
    befores = [r["context_before"] for r in result.rows if r["context_before"]]
    afters = [r["context_after"] for r in result.rows if r["context_before"]]
    rebuild = sum(float(r["rebuild_cost_usd"]) for r in result.rows)
    cols = st.columns(3)
    cols[0].metric("Compactions", f"{len(result.rows)}")
    median_before = median_after = 0
    if befores:
        median_before = int(pd.Series(befores).median())
        median_after = int(pd.Series(afters).median())
        drop = 100 * (1 - median_after / median_before) if median_before else 0
        cols[1].metric("Median context dropped", f"-{drop:,.0f}%")
    cols[2].metric("Total rebuild cost", f"${rebuild:,.2f}")
    if median_before:
        st.caption("Median context around a compaction:")
        echarts.render(
            _beforeafter_option(median_before, median_after), key="optimize_compact", height="150px"
        )
    with st.expander("Detail by compaction"):
        df = data.table_df(result)
        if "When (UTC)" in df.columns:
            df["When (UTC)"] = (
                pd.to_datetime(df["When (UTC)"], errors="coerce", utc=True)
                .dt.strftime("%Y-%m-%d %H:%M")
                .fillna("")
            )
        st.dataframe(
            df.style.format(
                {
                    "Context before": "{:,.0f}",
                    "Context after": "{:,.0f}",
                    "Reduction": "{:,.1f}%",
                    "Rebuild write": "{:,.0f}",
                    "Rebuild cost": "${:,.2f}",
                },
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )
        for note in _notes_after_source(result):
            st.caption(note)


def main() -> None:
    """Render the Optimize page."""
    st.title("Optimize")
    # This page has no sidebar filters; keep any selection from other pages alive.
    filters.persist_filters()
    st.caption(
        "Prescriptive analyses on the **request grain** of the full extracted "
        "history — the global sidebar filters do not apply on this page."
    )

    ds = data.load_dataset()
    if not ds.requests:
        st.info(
            "No request-grain data: these analyses need `requests.csv`. "
            "Re-run `prompt-analytics extract` to produce the v2 schema."
        )
        st.stop()
    primary = data.primary_provider()

    # Headline uses default thresholds (50 / 30) for a stable estimate; card 2's
    # sliders let the reader explore without moving the top-line number.
    ttl = analytics.ttl_losses(ds, primary)
    rec_base = analytics.recommendations(ds, primary, min_prompts=50, compact_at=30)
    _render_headline(ttl, rec_base)

    _render_pauses(ttl, primary)
    _render_recommendations(ds, primary)
    _render_compactions(ds, primary)


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
