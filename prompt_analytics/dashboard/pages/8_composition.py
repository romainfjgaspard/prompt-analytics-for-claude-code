"""Composition page: where the cost goes, *by content* (the product's spine).

The dashboard has a backbone -- **input** (what you asked), **output** (what
Claude produced), **context** (what fills the cache it re-reads). This page is
the narrated home of that spine, read as one whole: three sections in cost order
(input -> output -> context), each with the same shape (a KPI row, a headline
chart, a drill), then a unified **Files** table that crosses output and context
per file.

* **Input** (categories) -- a cost-by-category recap of the Prompts page, so the
  spine starts here too (DASH1).
* **Output** (Axe C) -- language mix, code vs tests, cost per language, and the
  prose-vs-code split of the generated tokens: the differentiator cc-lens never
  filled in (its ``languages`` field stays empty) and never priced.
* **Context** (Axe D) -- what fills the cached, re-read context, split into the
  one-off **loading** and the **rent** paid every turn it lingers; the attributed
  total reconciles to the billed cache cost to the dollar.
* **Files** (DASH4) -- one row per file crossing edits + line diff (output) with
  reads + context cost (loading + rent): a file's whole cost of ownership.

Every section is a *view* over the same analytics the CLI prints
(:func:`analytics.by_category` / :func:`output_composition` / :func:`context_cost`
/ :func:`file_footprint`), narrowed to the global sidebar / chart-click selection
via :func:`analytics.filter_prompt_ids` so it honours the same filter as every
other tab. Read-only: language/file are not global filter dimensions, so the
charts emit no cross-filter. Metrics only -- no source code is ever read here.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics import analytics
from prompt_analytics.context import NO_LANGUAGE
from prompt_analytics.dashboard import data, echarts, filters, theme

# How many languages to show before folding the long tail into an "Other" slice.
_MIX_TOP = 12
_COST_TOP = 10
# How many rows the categorical bars (categories, context elements) show.
_CAT_TOP = 12
_CTX_TOP = 12

# The four-bucket Axe D taxonomy: human labels for the context sources.
_CTX_LABELS = {
    "conversation": "Conversation",
    "file_read": "Files read",
    "tool_output": "Tool output",
    "config": "Config / setup",
}
# Loading is a one-off cache *write*, rent a repeated cache *read*: colour them
# with the token palette so the mechanic is legible (purple write, green read).
_LOAD_COLOR = theme.TOKEN_TYPE_COLORS["cache_write_1h"]
_RENT_COLOR = theme.TOKEN_TYPE_COLORS["cache_read"]


def _fold_other(
    pairs: list[tuple[str, float]], top: int, other_label: str = "Other languages"
) -> list[tuple[str, float]]:
    """Keep the ``top`` largest pairs (by value), summing the rest into one bucket."""
    ordered = sorted(pairs, key=lambda kv: -kv[1])
    head = ordered[:top]
    tail_sum = sum(v for _, v in ordered[top:])
    if tail_sum > 0:
        head.append((other_label, tail_sum))
    return head


def _hbar(
    title: str,
    names: list[str],
    values: list[float],
    colors: list[str],
    labels: list[str],
    *,
    money: bool = True,
) -> dict[str, Any]:
    """A horizontal bar with a per-bar text label (values shown, not just hidden).

    Reading magnitudes off a labeled bar is far clearer than a donut whose only
    on-chart text is a percentage -- here every bar carries its own ``$`` figure
    on the right. Largest at the top (inverse category axis). Shared across the
    page so the three sections look like one family.
    """
    c = echarts.colors()
    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": title,
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 8, "right": 160, "top": 48, "bottom": 24, "containLabel": True}
    option["tooltip"].update({"trigger": "item", "formatter": "{b}: ${c}" if money else "{b}: {c}"})
    xaxis = echarts.value_axis(money=money)
    xaxis["max"] = round(max(values) * 1.3, 2) if values else 1
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.category_axis(names, inverse=True)
    option["series"] = [
        {
            "type": "bar",
            "data": [
                {"value": v, "itemStyle": {"color": col}}
                for v, col in zip(values, colors, strict=True)
            ],
            "itemStyle": {"borderRadius": [0, 4, 4, 0]},
            "label": {
                "show": True,
                "position": "right",
                "color": c["text"],
                "formatter": echarts.js(
                    "function(p){var L=" + json.dumps(labels) + ";return L[p.dataIndex];}"
                ),
            },
        }
    ]
    return option


# ---------------------------------------------------------------------------
# Input section (categories) -- the spine starts here too (DASH1).
# ---------------------------------------------------------------------------


def _category_cost_option(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Horizontal bar of cost per category (the spend split of what you asked)."""
    priced = [(str(r["category"]), float(r["cost_usd"] or 0.0)) for r in rows]
    priced = [(name, value) for name, value in priced if value > 0]
    if not priced:
        return None
    priced = sorted(priced, key=lambda kv: -kv[1])[:_CAT_TOP]
    names = [name for name, _ in priced]
    values = [round(value, 2) for _, value in priced]
    colors = [theme.CATEGORY_COLORS.get(name, theme.PALETTE[7]) for name in names]
    labels = [f"${v:,.2f}" for v in values]
    return _hbar("Where the spend goes, by category", names, values, colors, labels)


def _render_input_section(ds: analytics.Dataset, provider: str) -> None:
    """Cost-by-category recap of the Prompts page (the input half of the spine)."""
    table = analytics.by_category(ds, provider)
    rows = [r for r in table.rows if str(r.get("category")) != "TOTAL"]
    categorized = [r for r in rows if str(r.get("category")) != "(uncategorized)"]
    if not categorized:
        st.info(
            "No categorized prompts in range. Run `prompt-analytics categorize` to fill the "
            "**input** breakdown (the full view lives on the **Prompts** tab)."
        )
        return

    total_cost = sum(float(r["cost_usd"] or 0.0) for r in rows)
    total_prompts = sum(int(r["prompts"] or 0) for r in rows)
    top = max(categorized, key=lambda r: float(r["cost_usd"] or 0.0))

    cols = st.columns(4)
    cols[0].metric("Prompts", f"{total_prompts:,}")
    cols[1].metric("Categories", f"{len(categorized):,}")
    cols[2].metric("Top category", str(top["category"]).title())
    cols[3].metric(
        f"Input cost ({provider})",
        f"${total_cost:,.2f}",
        delta=f"{float(top['cost_share_pct'] or 0):.0f}% {top['category']}",
        delta_color="off",
    )

    option = _category_cost_option(rows)
    if option is not None:
        echarts.render(option, key="comp_category_cost", height="360px")
    st.caption(
        "What you **asked for**, priced by category. The full breakdown — prompt counts, "
        "complexity, cost per prompt — lives on the **Prompts** tab; here it anchors the "
        "left end of the spine (input → output → context)."
    )


# ---------------------------------------------------------------------------
# Output section (Axe C) -- what Claude produced.
# ---------------------------------------------------------------------------


def _language_mix_option(comp: analytics.OutputComposition) -> dict[str, Any] | None:
    """Horizontal stacked bar of lines produced per language, split Code vs Tests.

    Answers "language mix" and "code vs tests" in one chart: bar length is the
    lines added in that language, the stack shows how much of it is tests.
    """
    langs = [lng for lng in comp.languages if lng.lines_added > 0]
    if not langs:
        return None
    langs = sorted(langs, key=lambda x: -x.lines_added)[:_MIX_TOP]
    names = [lng.language for lng in langs]
    code_lines = [int(lng.lines_added - lng.test_added) for lng in langs]
    test_lines = [int(lng.test_added) for lng in langs]

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Language mix — lines written",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "roundRect"}
    option["grid"] = {"left": 8, "right": 56, "top": 48, "bottom": 40, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["xAxis"] = echarts.value_axis(name="Lines added")
    option["yAxis"] = echarts.category_axis(names, inverse=True)
    option["series"] = [
        {
            "name": "Code",
            "type": "bar",
            "stack": "lines",
            "data": code_lines,
            "itemStyle": {"color": theme.PALETTE[1]},
        },
        {
            "name": "Tests",
            "type": "bar",
            "stack": "lines",
            "data": test_lines,
            "itemStyle": {"color": theme.CATEGORY_COLORS["test"], "borderRadius": [0, 4, 4, 0]},
        },
    ]
    return option


def _prose_vs_code_option(comp: analytics.OutputComposition) -> dict[str, Any] | None:
    """Two labeled bars: the prose half vs the code half of the output spend.

    The headline of the section -- you see *both* parts of the output and the
    dollars next to each (not a percentage hidden behind a hover). Prose on top
    (cyan, "what comes out"), code below (coral).
    """
    prose, code = round(comp.prose_cost, 2), round(comp.code_cost, 2)
    if prose <= 0 and code <= 0:
        return None
    total = prose + code
    names = ["Prose / explanation", "Code / tooling"]
    values = [prose, code]
    colors = [theme.TOKEN_TYPE_COLORS["output"], theme.PALETTE[0]]
    tokens = [comp.prose_tokens, comp.code_tokens]
    labels = [
        f"${v:,.2f}  ·  {pct:.0f}%  ·  {tok:,} tok"
        for v, tok, pct in zip(
            values, tokens, [100 * v / total if total else 0 for v in values], strict=True
        )
    ]
    return _hbar(f"Output spend — prose vs code ({comp.provider})", names, values, colors, labels)


def _cost_by_language_option(comp: analytics.OutputComposition) -> dict[str, Any] | None:
    """Horizontal bars of the estimated output spend by language (code side) + tooling."""
    pairs: list[tuple[str, float]] = [
        (lng.language, round(lng.code_cost, 4)) for lng in comp.languages if lng.code_cost > 0
    ]
    if comp.tooling_cost > 0:
        pairs.append(("(other tooling)", round(comp.tooling_cost, 4)))
    if not pairs:
        return None
    folded = _fold_other(pairs, _COST_TOP)
    names = [name for name, _ in folded]
    values = [round(value, 2) for _, value in folded]
    color_map = theme.language_color_map(names)
    colors = [color_map.get(name, "#9CA3AF") for name in names]
    labels = [f"${v:,.2f}" for v in values]
    return _hbar(
        f"Where the code spend went, by language ({comp.provider})", names, values, colors, labels
    )


def _render_output_headline(comp: analytics.OutputComposition) -> None:
    """Four KPI cards summarizing what the assistant produced."""
    gen_cost = comp.prose_cost + comp.code_cost
    code_share = round(100 * comp.code_cost / gen_cost, 1) if gen_cost else 0.0
    test_pct = round(100 * comp.total_test / comp.total_added, 1) if comp.total_added else 0.0

    cols = st.columns(4)
    cols[0].metric(
        "Lines written (+)",
        f"{comp.total_added:,}",
        delta=f"−{comp.total_deleted:,} removed" if comp.total_deleted else None,
        delta_color="off",
    )
    cols[1].metric("Files touched", f"{comp.total_files:,}")
    cols[2].metric("Output that is tests", f"{test_pct:.0f}%")
    cols[3].metric(
        f"Generation cost ({comp.provider})",
        f"${gen_cost:,.2f}",
        delta=f"{code_share:.0f}% code",
        delta_color="off",
    )


def _render_output_section(comp: analytics.OutputComposition) -> None:
    """The Axe-C output composition: prose/code headline + language drill."""
    _render_output_headline(comp)

    prose_code = _prose_vs_code_option(comp)
    if prose_code is not None:
        echarts.render(prose_code, key="comp_prose_vs_code", height="220px")
        gen_cost = comp.prose_cost + comp.code_cost
        st.caption(
            f"Every output message is part **explanation** (its text) and part "
            f"**code/tooling** (its tool calls). Of the **${gen_cost:,.2f}** spent "
            f"generating output, here is the split. _Estimate:_ each prompt's real "
            f"output cost is prorated by a local tokenizer's prose/code token weight."
        )
    else:
        st.info("No generated-output tokens in range.")

    st.subheader("Inside the code")
    left, right = st.columns(2)
    with left:
        cost_lang = _cost_by_language_option(comp)
        if cost_lang is not None:
            echarts.render(cost_lang, key="comp_cost_by_language", height="420px")
            st.caption(
                f"The **${comp.code_cost:,.2f}** code half, attributed to each language by "
                "line churn (`(other tooling)` = code tokens spent on Bash/Read/etc., no "
                "file edited)."
            )
        else:
            st.info("No code spend to attribute to a language in range.")
    with right:
        mix = _language_mix_option(comp)
        if mix is not None:
            echarts.render(mix, key="comp_language_mix", height="420px")
            st.caption(
                f"Exact line diffs: {comp.total_added:,} lines added across "
                f"{comp.total_files:,} files · {comp.total_test:,} in tests"
                + ("" if len(comp.languages) <= _MIX_TOP else f" · top {_MIX_TOP} shown")
            )
        else:
            st.info("No file-edit metrics in range (prose only, or read-only tools).")

    st.caption("👉 The same breakdown on the command line: `prompt-analytics by-output`.")


# ---------------------------------------------------------------------------
# Context section (Axe D) -- what fills the cached, re-read context.
# ---------------------------------------------------------------------------


def _context_label(element: analytics.ContextElementCost) -> str:
    """Human label for a context element: its source, plus the file language."""
    base = _CTX_LABELS.get(element.source, element.source)
    return f"{base} · {element.language}" if element.language != NO_LANGUAGE else base


def _load_vs_rent_option(comp: analytics.ContextCost) -> dict[str, Any] | None:
    """Two labeled bars: the one-off loading vs the rent of the whole context."""
    load, rent = round(comp.load_cost, 2), round(comp.rent_cost, 2)
    if load <= 0 and rent <= 0:
        return None
    total = load + rent
    names = ["Loading (one-off)", "Rent (every turn)"]
    values = [load, rent]
    colors = [_LOAD_COLOR, _RENT_COLOR]
    labels = [f"${v:,.2f}  ·  {(100 * v / total if total else 0):.0f}%" for v in values]
    return _hbar(
        f"Context spend — loading vs rent ({comp.provider})", names, values, colors, labels
    )


def _context_breakdown_option(comp: analytics.ContextCost) -> dict[str, Any] | None:
    """Horizontal stacked bar of the top context elements, split loading vs rent."""
    elements = [e for e in comp.elements if e.total_cost > 0][:_CTX_TOP]
    if not elements:
        return None
    names = [_context_label(e) for e in elements]
    load = [round(e.load_cost, 4) for e in elements]
    rent = [round(e.rent_cost, 4) for e in elements]

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Top context cost — what lingers",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "roundRect"}
    option["grid"] = {"left": 8, "right": 80, "top": 48, "bottom": 40, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    xaxis = echarts.value_axis(money=True, name="USD")
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.category_axis(names, inverse=True)
    option["series"] = [
        {
            "name": "Loading",
            "type": "bar",
            "stack": "ctx",
            "data": load,
            "itemStyle": {"color": _LOAD_COLOR},
        },
        {
            "name": "Rent",
            "type": "bar",
            "stack": "ctx",
            "data": rent,
            "itemStyle": {"color": _RENT_COLOR, "borderRadius": [0, 4, 4, 0]},
        },
    ]
    return option


def _render_context_section(comp: analytics.ContextCost) -> None:
    """The Axe-D context cost: loading-vs-rent headline + top-elements drill."""
    total = comp.total_cost
    rent_share = round(100 * comp.rent_cost / total, 1) if total else 0.0
    unattr_share = round(100 * comp.unattributed_cost / total, 1) if total else 0.0

    cols = st.columns(4)
    cols[0].metric(f"Context cost ({comp.provider})", f"${total:,.2f}")
    cols[1].metric(
        "Rent (re-read each turn)",
        f"${comp.rent_cost:,.2f}",
        delta=f"{rent_share:.0f}% of cache",
        delta_color="off",
    )
    cols[2].metric("Loading (one-off)", f"${comp.load_cost:,.2f}")
    cols[3].metric(
        "Unattributed",
        f"${comp.unattributed_cost:,.2f}",
        delta=f"{unattr_share:.0f}% of total",
        delta_color="off",
    )

    headline = _load_vs_rent_option(comp)
    if headline is not None:
        echarts.render(headline, key="comp_load_vs_rent", height="220px")
        st.caption(
            f"Context is cached once (**loading**, a cache *write*) then paid again every "
            f"turn it stays (**rent**, a cache *read* = size × turns of presence). "
            f"**{rent_share:.0f}%** of the cache bill is rent — the cost of context that "
            f"lingers, which `/compact` and a leaner CLAUDE.md cut."
        )

    breakdown = _context_breakdown_option(comp)
    if breakdown is not None:
        echarts.render(breakdown, key="comp_context_breakdown", height="420px")
        if comp.elements:
            top = comp.elements[0]
            st.caption(
                f"Biggest context cost: **{_context_label(top)}** at ${top.total_cost:,.2f} "
                f"(${top.load_cost:,.2f} load + ${top.rent_cost:,.2f} rent). The attributed "
                f"total reconciles to the billed cache cost to the dollar; the "
                f"{unattr_share:.0f}% unattributed is cache on turns with no measured element "
                f"(post-compaction summaries — the parentUuid ≈ API-context caveat)."
            )
    else:
        st.info("No per-element context cost in range.")

    st.caption("👉 The same breakdown on the command line: `prompt-analytics by-context`.")


# ---------------------------------------------------------------------------
# Files section (DASH4) -- one row per file, output edits crossed with reads.
# ---------------------------------------------------------------------------

_FILE_HEADERS = {
    "path": "File",
    "language": "Language",
    "kind": "Kind",
    "edits": "Edits",
    "lines_added": "Lines +",
    "lines_deleted": "Lines −",
    "reads": "Reads",
    "load_usd": "Load $",
    "rent_usd": "Rent $",
    "context_usd": "Context $",
}


def _render_files_section(ds: analytics.Dataset, provider: str) -> None:
    """The unified per-file footprint: a sortable/filterable Explorer-style table."""
    table = analytics.file_footprint(ds, provider)
    if not table.rows:
        st.info(
            "No per-file data yet. The file identity (relative path) ships with the latest "
            "extractor — re-run `prompt-analytics extract`, then revisit this page."
        )
        return

    df = pd.DataFrame(table.rows).rename(columns=_FILE_HEADERS)

    controls = st.columns([3, 2])
    query = controls[0].text_input(
        "Filter files",
        key="comp_file_filter",
        placeholder="path or language…",
        label_visibility="collapsed",
    )
    scope = controls[1].selectbox(
        "Footprint",
        ["All files", "Edited", "Read but never edited"],
        key="comp_file_scope",
        label_visibility="collapsed",
    )

    view = df
    if query:
        q = query.strip().lower()
        view = view[
            view["File"].str.lower().str.contains(q, regex=False)
            | view["Language"].str.lower().str.contains(q, regex=False)
        ]
    if scope == "Edited":
        view = view[view["Edits"] > 0]
    elif scope == "Read but never edited":
        view = view[(view["Edits"] == 0) & (view["Reads"] > 0)]

    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        column_config={
            "File": st.column_config.TextColumn("File", width="large"),
            "Edits": st.column_config.NumberColumn("Edits", format="%d"),
            "Lines +": st.column_config.NumberColumn("Lines +", format="%d"),
            "Lines −": st.column_config.NumberColumn("Lines −", format="%d"),
            "Reads": st.column_config.NumberColumn("Reads", format="%d"),
            "Load $": st.column_config.NumberColumn("Load $", format="$%.2f"),
            "Rent $": st.column_config.NumberColumn("Rent $", format="$%.2f"),
            "Context $": st.column_config.NumberColumn("Context $", format="$%.2f"),
        },
    )

    edited = int((df["Edits"] > 0).sum())
    read_only = int(((df["Edits"] == 0) & (df["Reads"] > 0)).sum())
    st.caption(
        f"One row per file: **edits** + line diff (output) crossed with **reads** + the "
        f"**context cost** they drove (loading + rent). {len(df):,} files — {edited:,} edited, "
        f"**{read_only:,} read but never edited** (pure context cost, the first candidates to "
        f"keep out of context). Sort by any column; metrics only — relative paths, never "
        f"content. 👉 On the command line: `prompt-analytics by-file`."
    )


# ---------------------------------------------------------------------------
# Page.
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Composition page: input -> output -> context, then Files."""
    st.title("Composition")
    st.caption(
        "Where your cost goes, **by content** — read as one spine: **input** (what you "
        "ask) → **output** (what Claude produces) → **context** (what fills the cache it "
        "re-reads), then a per-file footprint that crosses output and context."
    )

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all)

    prompts = frames.get("prompts", pd.DataFrame())
    primary = data.primary_provider()
    if prompts.empty or "prompt_id" not in prompts.columns:
        st.info("No data for the current filters.")
        st.stop()

    # The composition CSVs are not part of the dashboard frames; load the raw
    # dataset and narrow it to the same prompts the global filters kept (this
    # also narrows the session-grained context rows to those sessions).
    kept_ids = set(prompts["prompt_id"])
    ds = analytics.filter_prompt_ids(data.load_dataset(), kept_ids)
    comp = analytics.output_composition(ds, primary)
    ctx = analytics.context_cost(ds, primary)

    composition_ready = (
        comp.has_data or ctx.has_data or bool(ds.output_files) or bool(ds.context_cost)
    )
    if not composition_ready:
        st.info(
            "No composition data yet. The output (language mix, code vs tests, prose vs code) "
            "and context (loading vs rent, per-file cost) metrics ship with the latest "
            "extractor — re-run `prompt-analytics extract`, then revisit this page."
        )
        st.stop()

    theme.section(
        "Input — what you asked for",
        "The spend split of your prompts, by category (full detail on the Prompts tab).",
    )
    _render_input_section(ds, primary)

    theme.section(
        "Output — what Claude produced",
        "Prose vs code, the language mix, and the code/test split of what was written.",
    )
    if comp.has_data:
        _render_output_section(comp)
    else:
        st.info("No output-composition data in range.")

    theme.section(
        "Context — what fills the cache it re-reads",
        "What is cached and paid again every turn: the one-off loading vs the rent it pays.",
    )
    if ctx.has_data:
        _render_context_section(ctx)
    else:
        st.info("No context-composition data in range.")

    theme.section(
        "Files — every file's total footprint",
        "One row per file: its edits and line diff (output) crossed with its reads and "
        "context cost (loading + rent) — the file's whole cost of ownership.",
    )
    _render_files_section(ds, primary)


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
