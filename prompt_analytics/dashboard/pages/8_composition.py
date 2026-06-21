"""Composition page: where the cost goes, *by content* (the product's spine).

The dashboard has a backbone -- **input** (what you asked), **output** (what
Claude produced), **context** (what fills the cache it re-reads). This page is
the narrated home of that spine, read as one whole: three sections in cost order
(input -> output -> context), each with the same shape (a KPI row, a headline
chart, a drill), then a unified **Files** table that crosses output and context
per file, and finally the **Tasks** graph that lifts the spine to the unit of
work.

* **Input** (categories) -- a cost-by-category recap of the Prompts page, so the
  spine starts here too (DASH1).
* **Output** (Axe C) -- the prose-vs-code split of the generated spend and the
  language mix (code side) of what was written: the differentiator cc-lens never
  filled in (its ``languages`` field stays empty) and never priced.
* **Context** (Axe D) -- what fills the cached, re-read context, split into the
  one-off **loading** and the **rent** paid every turn it lingers; the attributed
  total reconciles to the billed cache cost to the dollar.
* **Files** (DASH4) -- one row per file crossing edits + line diff (output) with
  reads + context cost (loading + rent): a file's whole cost of ownership.
* **Tasks** (Axe B2 / DASH5) -- the cost graph: tasks as centres of gravity
  (size = cost, hue = dominant category) with their prompts orbiting as
  satellites, drillable per task via an ECharts force layout. The unit of work,
  the most telling level of the spine.

Every section is a *view* over the same analytics the CLI prints
(:func:`analytics.by_category` / :func:`output_composition` / :func:`context_cost`
/ :func:`file_footprint` / :func:`task_graph`), narrowed to the global sidebar /
chart-click selection
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
from prompt_analytics.dashboard import data, echarts, filters, impact, theme

# How many languages to show before folding the long tail into an "Other" slice.
_MIX_TOP = 12
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

# Char-count histogram buckets for the prompt-length distribution. Fixed edges
# (not data-derived) so the bar labels stay stable and a click maps to a known
# range; the top bucket is open-ended (``hi`` is ``None``). The label is the
# bridge between the chart (axis text) and the cross-filter (the clicked range).
_CHAR_BINS: list[tuple[int, int | None, str]] = [
    (0, 100, "0–100"),
    (100, 250, "100–250"),
    (250, 500, "250–500"),
    (500, 1000, "500–1k"),
    (1000, 2000, "1k–2k"),
    (2000, 5000, "2k–5k"),
    (5000, None, "5k+"),
]
_CHAR_RANGE_BY_LABEL = {label: (lo, hi) for lo, hi, label in _CHAR_BINS}


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


def _char_distribution_option(prompts: pd.DataFrame) -> dict[str, Any] | None:
    """Vertical histogram of prompt length in characters; emits ``XF_CHAR_BUCKET``.

    Reads ``char_count`` (native on every prompt) and bins it on the fixed
    :data:`_CHAR_BINS` edges so a clicked bar maps back to a known range. A
    companion to the category bar: together they answer "what you asked, by
    intent and by length".
    """
    if "char_count" not in prompts.columns:
        return None
    cc = pd.to_numeric(prompts["char_count"], errors="coerce").dropna()
    if cc.empty:
        return None
    counts: list[int] = []
    for lo, hi, _label in _CHAR_BINS:
        mask = cc >= lo
        if hi is not None:
            mask &= cc < hi
        counts.append(int(mask.sum()))
    labels = [label for _, _, label in _CHAR_BINS]

    c = echarts.colors()
    option = echarts.base_option(color=[theme.PALETTE[1]])
    option["legend"] = {"show": False}
    option["title"] = {
        "text": "How long your prompts are (characters)",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 64, "right": 24, "top": 64, "bottom": 56, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["xAxis"] = echarts.category_axis(labels)
    option["xAxis"]["name"] = "Characters"
    option["xAxis"]["nameLocation"] = "middle"
    option["xAxis"]["nameGap"] = 32
    option["yAxis"] = echarts.value_axis(name="Prompts")
    option["yAxis"]["nameLocation"] = "middle"
    option["yAxis"]["nameGap"] = 46
    option["series"] = [
        {
            "type": "bar",
            "data": counts,
            "itemStyle": {"color": theme.PALETTE[1], "borderRadius": [4, 4, 0, 0]},
            "label": {"show": True, "position": "top", "color": c["text"]},
        }
    ]
    return option


def _apply_char_click(value: Any) -> None:
    """Clicked char-histogram bar -> narrow the dashboard to that length bucket.

    The clicked value is the bar's label (e.g. ``"500–1k"``); it is mapped back to
    its ``[lo, hi]`` range and written to the :data:`filters.XF_CHAR_BUCKET` drill
    (badge + Reset, never the sidebar). Reruns only on a real change, so the
    component's sticky value cannot re-apply the same bucket every rerun.
    """
    if not isinstance(value, str):
        return
    rng = _CHAR_RANGE_BY_LABEL.get(value)
    if rng is None:
        return
    if filters.set_cross_filter(filters.XF_CHAR_BUCKET, [rng[0], rng[1]]):
        st.rerun()


def _render_input_section(ds: analytics.Dataset, provider: str, prompts: pd.DataFrame) -> None:
    """Cost-by-category recap of the Prompts page (the input half of the spine)."""
    table = analytics.by_category(ds, provider)
    rows = [r for r in table.rows if str(r.get("category")) != "TOTAL"]
    categorized = [r for r in rows if str(r.get("category")) != "(uncategorized)"]
    if not categorized:
        st.info("No categorized prompts in range. Run `prompt-analytics categorize` to fill it.")
        return

    total_prompts = len(prompts)
    # Fresh-input tokens / cost only (token type ``input``) -- NOT by_category's
    # cost_usd, which is the whole prompt (input + output + cache ≈ the bill).
    input_tokens = sum(r["token_count"] for r in ds.tokens if r["token_type"] == "input")
    fresh_input_cost = analytics.input_cost(ds, provider)

    cols = st.columns(4)
    cols[0].metric("Prompts", f"{total_prompts:,}")
    cols[1].metric("Tokens", f"{input_tokens:,}", delta="fresh input only", delta_color="off")
    cols[2].metric("Categories", f"{len(categorized):,}")
    cols[3].metric(
        f"Input cost ({provider})",
        f"${fresh_input_cost:,.2f}",
        delta="fresh input tokens only",
        delta_color="off",
    )

    left, right = st.columns([2, 3])
    with left:
        option = _category_cost_option(rows)
        if option is not None:
            clicked = echarts.render(option, key="comp_category_cost", height="360px", click=True)
            echarts.apply_click(
                clicked, filters.KEY_CATEGORIES, synthetic=frozenset({"(uncategorized)"})
            )
        else:
            st.info("No category spend to show in range.")
    with right:
        dist = _char_distribution_option(prompts)
        if dist is not None:
            clicked = echarts.render(dist, key="comp_char_dist", height="360px", click=True)
            _apply_char_click(clicked)
        else:
            st.info("No prompt-length data in range.")

    st.caption(
        "👆 Click a **category** bar to filter the dashboard to that intent, or a **length** "
        "bar to filter to prompts of that size."
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
        "text": "Language mix — code lines written",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "roundRect"}
    option["grid"] = {"left": 8, "right": 56, "top": 64, "bottom": 48, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["xAxis"] = echarts.value_axis()
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


def _prose_vs_code_pie(comp: analytics.OutputComposition) -> dict[str, Any] | None:
    """A donut of the prose half vs the code half of the output spend.

    The headline of the section: you see *both* parts of the output and the
    dollars + share of each. Prose is cyan ("what comes out"), code coral.
    """
    prose, code = round(comp.prose_cost, 2), round(comp.code_cost, 2)
    if prose <= 0 and code <= 0:
        return None

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": f"Output spend — prose vs code ({comp.provider})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "circle"}
    option["tooltip"] = {**option["tooltip"], "trigger": "item", "formatter": "{b}: ${c} ({d}%)"}
    option["series"] = [
        {
            "type": "pie",
            "radius": ["40%", "66%"],
            "center": ["50%", "50%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderColor": c["grid"], "borderWidth": 2},
            "label": {"color": c["text"], "formatter": "{b}\n${c} · {d}%"},
            "data": [
                {
                    "value": prose,
                    "name": "Prose",
                    "itemStyle": {"color": theme.TOKEN_TYPE_COLORS["output"]},
                },
                {
                    "value": code,
                    "name": "Code",
                    "itemStyle": {"color": theme.PALETTE[0]},
                },
            ],
        }
    ]
    return option


def _render_output_headline(comp: analytics.OutputComposition) -> None:
    """Four KPI cards summarizing what the assistant produced."""
    gen_cost = comp.prose_cost + comp.code_cost
    gen_tokens = comp.prose_tokens + comp.code_tokens
    code_share = round(100 * comp.code_cost / gen_cost, 1) if gen_cost else 0.0

    cols = st.columns(4)
    cols[0].metric("Files touched", f"{comp.total_files:,}")
    cols[1].metric("Lines written (+)", f"{comp.total_added:,}")
    cols[2].metric("Tokens", f"{gen_tokens:,}", delta="generated output", delta_color="off")
    cols[3].metric(
        f"Generation cost ({comp.provider})",
        f"${gen_cost:,.2f}",
        delta=f"{code_share:.0f}% code",
        delta_color="off",
    )


def _render_output_section(comp: analytics.OutputComposition) -> None:
    """The Axe-C output composition: KPI row + prose/code pie + language mix."""
    _render_output_headline(comp)

    left, right = st.columns([2, 3])
    with left:
        pie = _prose_vs_code_pie(comp)
        if pie is not None:
            echarts.render(pie, key="comp_prose_vs_code", height="400px")
            st.caption(
                "Output is part **explanation** (text) and part **code/tooling** (tool calls), "
                "split by a local tokenizer's prose/code weight."
            )
        else:
            st.info("No generated-output tokens in range.")
    with right:
        mix = _language_mix_option(comp)
        if mix is not None:
            echarts.render(mix, key="comp_language_mix", height="400px")
            st.caption(
                f"**Code side only** — prose isn't tracked by language. {comp.total_added:,} "
                f"lines added · {comp.total_test:,} in tests"
                + ("" if len(comp.languages) <= _MIX_TOP else f" · top {_MIX_TOP} shown")
            )
        else:
            st.info("No file-edit metrics in range (prose only, or read-only tools).")


# ---------------------------------------------------------------------------
# Context section (Axe D) -- what fills the cached, re-read context.
# ---------------------------------------------------------------------------


def _context_label(element: analytics.ContextElementCost) -> str:
    """Human label for a context element: its source, plus the file language."""
    base = _CTX_LABELS.get(element.source, element.source)
    return f"{base} · {element.language}" if element.language != NO_LANGUAGE else base


def _load_vs_rent_pie(comp: analytics.ContextCost) -> dict[str, Any] | None:
    """A donut of the one-off loading vs the rent of the whole context, in tokens.

    Tokens, not dollars (the dollar figure lives in the KPI row): the headline is
    the *size* of what is written once vs re-read every turn. Loading is a cache
    write (purple), rent a cache read (green) -- the token palette makes the
    mechanic legible.
    """
    load, rent = comp.load_tokens, comp.rent_read_tokens
    if load <= 0 and rent <= 0:
        return None

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Context size — loading vs rent (tokens)",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "circle"}
    option["tooltip"] = {**option["tooltip"], "trigger": "item", "formatter": "{b}: {c} ({d}%)"}
    option["series"] = [
        {
            "type": "pie",
            "radius": ["40%", "66%"],
            "center": ["50%", "50%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderColor": c["grid"], "borderWidth": 2},
            "label": {"color": c["text"], "formatter": "{b}\n{d}%"},
            "data": [
                {
                    "value": load,
                    "name": "Loading (one-off)",
                    "itemStyle": {"color": _LOAD_COLOR},
                },
                {
                    "value": rent,
                    "name": "Rent (every turn)",
                    "itemStyle": {"color": _RENT_COLOR},
                },
            ],
        }
    ]
    return option


def _context_pareto_option(comp: analytics.ContextCost) -> dict[str, Any] | None:
    """Pareto of the top context elements by token size: bars + cumulative share.

    Tokens (not dollars): the bars are each element's total tokens (rent + load)
    descending, the line the running cumulative share of *all* attributed context
    tokens -- so the eye reads how few elements drive most of the context.
    """
    elements = [e for e in comp.elements if e.total_tokens > 0][:_CTX_TOP]
    if not elements:
        return None
    names = [_context_label(e) for e in elements]
    tokens = [e.total_tokens for e in elements]
    total = comp.total_tokens or 1
    cumulative: list[float] = []
    running = 0
    for t in tokens:
        running += t
        cumulative.append(round(100 * running / total, 1))

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Top context tokens — what lingers",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"show": False}
    # ``containLabel`` makes ``bottom`` reach to the *outer edge of the rotated
    # axis labels*, so a large bottom would leave that margin as empty canvas
    # below them -- keep it small and let containLabel reserve the label height.
    option["grid"] = {"left": 8, "right": 56, "top": 44, "bottom": 8, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    xaxis = echarts.category_axis(names)
    xaxis["axisLabel"] = {"color": c["text"], "rotate": 30, "interval": 0, "fontSize": 10}
    option["xAxis"] = xaxis
    # Tokens axis in millions (the raw counts run into the billions, so the long
    # ``1,000,000,000`` labels collided with the axis name); ``{c}M`` stays compact.
    left_y = echarts.value_axis(name="Tokens (millions)")
    left_y["nameLocation"] = "middle"
    left_y["nameGap"] = 52
    left_y["axisLabel"] = {
        "color": c["text"],
        "formatter": echarts.js("function(v){return (v/1e6).toLocaleString()+'M';}"),
    }
    option["yAxis"] = [
        left_y,
        {
            "type": "value",
            "name": "Cumulative %",
            "nameLocation": "middle",
            "nameGap": 44,
            "min": 0,
            "max": 100,
            "position": "right",
            "axisLabel": {"color": c["text"], "formatter": "{value}%"},
            "axisLine": {"lineStyle": {"color": c["axis"]}},
            "splitLine": {"show": False},
            "nameTextStyle": {"color": c["muted"]},
        },
    ]
    option["series"] = [
        {
            "name": "Tokens",
            "type": "bar",
            "data": tokens,
            "itemStyle": {"color": theme.PALETTE[5], "borderRadius": [4, 4, 0, 0]},
            "yAxisIndex": 0,
        },
        {
            "name": "Cumulative %",
            "type": "line",
            "data": cumulative,
            "yAxisIndex": 1,
            "smooth": False,
            "symbol": "circle",
            "symbolSize": 6,
            "lineStyle": {"color": theme.PALETTE[0], "width": 2},
            "itemStyle": {"color": theme.PALETTE[0]},
        },
    ]
    return option


def _top_context_source(comp: analytics.ContextCost) -> str | None:
    """The context source (conversation / files / …) with the highest cost."""
    by_source: dict[str, float] = {}
    for e in comp.elements:
        by_source[e.source] = by_source.get(e.source, 0.0) + e.total_cost
    if not by_source:
        return None
    return max(by_source.items(), key=lambda kv: kv[1])[0]


def _render_context_section(comp: analytics.ContextCost) -> None:
    """The Axe-D context cost: KPI row + loading-vs-rent pie + token Pareto."""
    total = comp.total_cost
    rent_share = round(100 * comp.rent_cost / total, 1) if total else 0.0
    top_source = _top_context_source(comp)
    top_source_label = _CTX_LABELS.get(top_source, top_source) if top_source else "—"

    cols = st.columns(4)
    cols[0].metric("Context tokens", f"{comp.total_tokens:,}")
    cols[1].metric("Rent share", f"{rent_share:.0f}%", delta="of context cost", delta_color="off")
    cols[2].metric("Top source", str(top_source_label))
    cols[3].metric(f"Context cost ({comp.provider})", f"${total:,.2f}")

    left, right = st.columns([2, 3])
    with left:
        pie = _load_vs_rent_pie(comp)
        if pie is not None:
            echarts.render(pie, key="comp_load_vs_rent", height="400px")
            st.caption(
                f"**Loading** is the one-off cache *write*; **rent** the cache *read* paid "
                f"every turn the context lingers. Rent is **{rent_share:.0f}%** of the cache "
                f"bill — the lingering-context cost you cut by compacting (`/compact`) and "
                f"keeping CLAUDE.md short."
            )
        else:
            st.info("No context tokens in range.")
    with right:
        pareto = _context_pareto_option(comp)
        if pareto is not None:
            echarts.render(pareto, key="comp_context_pareto", height="400px")
            if comp.elements:
                top = comp.elements[0]
                st.caption(
                    f"Biggest: **{_context_label(top)}** at {top.total_tokens:,} tokens "
                    f"(${top.total_cost:,.2f}). A few elements usually drive most of the context."
                )
        else:
            st.info("No per-element context size in range.")


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
        f"content."
    )


# ---------------------------------------------------------------------------
# Tasks section (Axe B2 / DASH5) -- the cost graph: tasks as centres, prompts
# as satellites, the most telling level of the "cost by content" spine.
# ---------------------------------------------------------------------------

# How many task centres the headline graph shows before the long tail is hidden
# (a force layout stays legible at a few dozen nodes, not hundreds).
_TASK_TOP = 40
# Task-node radius band (px), mapped from cost by a sqrt scale so area ~ cost.
_TASK_MIN_SIZE, _TASK_MAX_SIZE = 16.0, 56.0
# Prompt-satellite radius band (px), much smaller so centres read as centres.
_SAT_MIN_SIZE, _SAT_MAX_SIZE = 6.0, 16.0


def _task_color(category: str) -> str:
    """Category colour for a task/prompt node (shared with the Prompts page)."""
    return theme.CATEGORY_COLORS.get(category, theme.CATEGORY_COLORS["(uncategorized)"])


def _cat_label(category: str) -> str:
    """Legend-friendly category label (``(uncategorized)`` stays as-is)."""
    return category if category.startswith("(") else category.title()


def _scaled_size(value: float, vmax: float, lo: float, hi: float) -> float:
    """Map a cost to a radius on a sqrt scale (area ~ cost), clamped to [lo, hi]."""
    if vmax <= 0:
        return lo
    return float(round(lo + (hi - lo) * (max(value, 0.0) / vmax) ** 0.5, 1))


def _task_graph_option(
    graph: analytics.TaskGraph, focus: str | None = None
) -> dict[str, Any] | None:
    """ECharts force-layout graph: task centres (size = cost, hue = category) with
    their prompt satellites; ``focus`` narrows it to a single task and its prompts.

    Colour is driven by ``categories`` (the legend), so toggling a category in the
    legend dims every task and prompt of that intent at once. Tasks carry a label
    and a rich tooltip; satellites stay quiet (colour + hover only).
    """
    tasks = [t for t in graph.tasks if focus is None or t.task_id == focus]
    if not tasks:
        return None
    kept = {t.task_id for t in tasks}
    sats = [s for s in graph.satellites if s.task_id in kept]

    # Stable category -> legend index, in cost order so the legend reads top-down.
    cats: list[str] = []
    for cat in [t.category for t in tasks] + [s.category for s in sats]:
        if cat not in cats:
            cats.append(cat)
    cat_index = {cat: i for i, cat in enumerate(cats)}

    cost_max = max((t.cost for t in tasks), default=0.0)
    sat_max = max((s.cost for s in sats), default=0.0)
    # Focusing one task zooms in: give its centre and prompts more room.
    task_lo, task_hi = (_TASK_MIN_SIZE, _TASK_MAX_SIZE) if focus is None else (40.0, 90.0)
    sat_lo, sat_hi = (_SAT_MIN_SIZE, _SAT_MAX_SIZE) if focus is None else (12.0, 30.0)

    nodes: list[dict[str, Any]] = []
    for t in tasks:
        label = t.name if len(t.name) <= 28 else t.name[:27] + "…"
        nodes.append(
            {
                "id": t.task_id,
                "name": t.name,
                "kind": "task",
                "category": cat_index[t.category],
                "symbolSize": _scaled_size(t.cost, cost_max, task_lo, task_hi),
                "value": round(t.cost, 2),
                "cost": round(t.cost, 2),
                "prompts": t.prompts,
                "origin": t.origin,
                "ctx": t.context_pct,
                "itemStyle": {"color": _task_color(t.category), "borderColor": "#0B1220"},
                "label": {"show": True, "formatter": label},
            }
        )
    for s in sats:
        nodes.append(
            {
                "id": s.prompt_id,
                "name": s.category,
                "kind": "prompt",
                "category": cat_index[s.category],
                "symbolSize": _scaled_size(s.cost, sat_max, sat_lo, sat_hi),
                "value": round(s.cost, 2),
                "cost": round(s.cost, 2),
                "itemStyle": {"color": _task_color(s.category), "opacity": 0.85},
                "label": {"show": False},
            }
        )
    links = [{"source": s.task_id, "target": s.prompt_id} for s in sats]

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Cost by task — centres of gravity",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {
        "data": [_cat_label(cat) for cat in cats],
        "bottom": 0,
        "textStyle": {"color": c["text"]},
        "icon": "circle",
        "type": "scroll",
    }
    option["tooltip"] = {
        **option["tooltip"],
        "trigger": "item",
        "formatter": echarts.js(
            "function(p){if(p.dataType==='edge'){return '';}var d=p.data;"
            "if(d.kind==='task'){return '<b>'+d.name+'</b><br/>'+d.origin+' task"
            " · '+d.prompts+' prompts<br/>$'+Number(d.cost).toFixed(2)+' · '"
            "+Number(d.ctx).toFixed(0)+'% context';}"
            "return 'prompt · $'+Number(d.cost).toFixed(2);}"
        ),
    }
    option["series"] = [
        {
            "type": "graph",
            "layout": "force",
            "roam": True,
            "draggable": True,
            "categories": [
                {"name": _cat_label(cat), "itemStyle": {"color": _task_color(cat)}} for cat in cats
            ],
            "force": {
                "repulsion": 140 if focus is None else 320,
                "edgeLength": [20, 60],
                "gravity": 0.12,
            },
            "label": {"position": "right", "color": c["text"], "fontSize": 11},
            "lineStyle": {"color": c["axis"], "opacity": 0.45, "width": 1},
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 2}},
            "data": nodes,
            "links": links,
        }
    ]
    return option


def _render_tasks_section(graph: analytics.TaskGraph) -> None:
    """The Axe-B2 task graph: KPI row, a focus picker, then the force-layout graph."""
    shown = graph.tasks
    top_task = shown[0] if shown else None

    cols = st.columns(4)
    cols[0].metric("Tasks", f"{graph.total_tasks:,}")
    cols[1].metric(
        "From the TodoWrite spine",
        f"{graph.todo_tasks:,}",
        delta=f"{graph.total_tasks - graph.todo_tasks:,} inferred",
        delta_color="off",
    )
    cols[2].metric(
        "Most expensive task",
        f"${top_task.cost:,.2f}" if top_task else "—",
        delta=(top_task.name[:22] if top_task else None),
        delta_color="off",
    )
    cols[3].metric(
        f"Task cost ({graph.provider})",
        f"${graph.grand_total:,.2f}",
        delta=(
            f"{round(100 * graph.context_total / graph.grand_total):.0f}% context"
            if graph.grand_total
            else None
        ),
        delta_color="off",
    )

    # Drill by task: the headline shows the top centres, the picker zooms one in.
    labels = {f"{t.name[:48]}  ·  ${t.cost:,.2f}": t.task_id for t in shown}
    choice = st.selectbox(
        "Focus on a task",
        ["All tasks (top centres)", *labels.keys()],
        key="comp_task_focus",
        label_visibility="collapsed",
    )
    focus = labels.get(choice)

    option = _task_graph_option(graph, focus)
    if option is not None:
        echarts.render(option, key="comp_task_graph", height="560px")

    if focus is None:
        st.caption(
            f"Each **task** is a centre of gravity (its size is its cost, its colour the "
            f"dominant category); the **prompts** that served it orbit around it. Top "
            f"{min(len(shown), _TASK_TOP)} of {graph.total_tasks:,} tasks shown — pick one above "
            f"to zoom into its prompts. Drag nodes, scroll to zoom, toggle a category in the "
            f"legend."
        )
    else:
        focused = next(t for t in shown if t.task_id == focus)
        st.caption(
            f"**{focused.name}** — {focused.origin} task, {focused.prompts:,} prompts, "
            f"**${focused.cost:,.2f}** ({focused.context_pct:.0f}% context), dominant category "
            f"_{focused.category}_. Each satellite is one prompt, coloured by its own intent."
        )


def _render_task_side(graph: analytics.TaskGraph, *, side: str, pivot: str, key: str) -> None:
    """One side (before / after) of the task graph in compare mode: KPI + force graph."""
    when = f"before {pivot}" if side == "Before" else f"from {pivot}"
    st.markdown(f"**{side}** · _{when}_")
    if not graph.has_data:
        st.info("No task data on this side of the pivot.")
        return
    cols = st.columns(2)
    cols[0].metric("Tasks", f"{graph.total_tasks:,}")
    cols[1].metric(
        f"Task cost ({graph.provider})",
        f"${graph.grand_total:,.2f}",
        delta=(
            f"${graph.grand_total / graph.total_tasks:,.2f}/task" if graph.total_tasks else None
        ),
        delta_color="off",
    )
    option = _task_graph_option(graph)
    if option is not None:
        echarts.render(option, key=key, height="460px")


def _render_tasks_comparison(ds: analytics.Dataset, provider: str, pivot: str) -> None:
    """Before/after of the B2 task graph (DASH2): the constellation on each side.

    The cost-per-task headline above the graphs is the workload-normalized lens;
    the two force layouts show how the constellation of work itself shifted across
    the switch date.
    """
    before_ds, after_ds = analytics.split_on_pivot(ds, pivot)
    before = analytics.task_graph(before_ds, provider, top=_TASK_TOP)
    after = analytics.task_graph(after_ds, provider, top=_TASK_TOP)

    left, right = st.columns(2)
    with left:
        _render_task_side(before, side="Before", pivot=pivot, key="comp_task_graph_before")
    with right:
        _render_task_side(after, side="After", pivot=pivot, key="comp_task_graph_after")
    st.caption(
        "Each **task** is a centre of gravity (size = cost, colour = dominant category), its "
        "**prompts** orbiting as satellites — shown before vs after your switch date. The "
        "cost-per-task figures normalize for how much work each side carried."
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

    # Compare mode (Axe E / DASH2): lead with the before/after panel — its Output
    # cost share and Context rent share deltas are exactly the composition story.
    pivot = impact.current_pivot()
    if pivot is not None:
        theme.section(
            f"Impact of {pivot} — before vs after",
            "The before/after of your switch date across the whole spine, in workload-normalized "
            "ratios (output cost share and context rent share are the composition deltas).",
        )
        impact.render_impact_panel(ds, primary, pivot)

    theme.section(
        "Input — what you asked for",
        "The spend split of your prompts, by category and by length.",
    )
    _render_input_section(ds, primary, prompts)

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

    theme.section(
        "Tasks — the cost graph",
        "The unit of work, not the prompt: tasks as centres of gravity (size = cost, "
        "colour = dominant category), the prompts that served each one orbiting around it.",
    )
    if pivot is not None:
        # Compare mode: the constellation before vs after the switch date.
        _render_tasks_comparison(ds, primary, pivot)
    else:
        graph = analytics.task_graph(ds, primary, top=_TASK_TOP)
        if graph.has_data:
            _render_tasks_section(graph)
        else:
            st.info(
                "No task data yet. Task attribution (the TodoWrite spine + inference) ships with "
                "the latest extractor — re-run `prompt-analytics extract`, then revisit this page."
            )


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
