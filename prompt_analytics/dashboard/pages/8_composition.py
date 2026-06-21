"""Composition page: where the cost goes, *by content* (the product's spine).

The dashboard has a backbone -- **input** (what you asked), **output** (what
Claude produced), **context** (what fills the cache it re-reads). This page is
the narrated home of that spine, read as one whole: three sections in cost order
(input -> output -> context), each with the same shape (a KPI row, a headline
chart, a drill), and finally the **Files** graph that lifts the spine to what the
work actually touched. The per-file search + drill lives on its own **File
Explorer** page (a table); this page is the overview constellation.

* **Input** (categories) -- the cost-by-category breakdown of what you asked, so
  the spine starts here too (DASH1; this absorbed the old Prompts tab).
* **Output** (Axe C) -- the prose-vs-code split of the generated spend and the
  language mix (code side) of what was written: the differentiator cc-lens never
  filled in (its ``languages`` field stays empty) and never priced.
* **Context** (Axe D) -- what fills the cached, re-read context, split into the
  one-off **loading** and the **rent** paid every turn it lingers; the attributed
  total reconciles to the billed cache cost to the dollar.
* **Files** (Axe C+D / DASH5) -- the cost graph: files as centres of gravity
  (size = context cost, hue = language) with the prompts that edited them orbiting
  as satellites, drillable per file via an ECharts force layout. A real, named,
  every-user identity -- it replaced the task constellation, whose inferred task
  names read as noise once measured on real data.

Every section is a *view* over the same analytics the CLI prints
(:func:`analytics.by_category` / :func:`output_composition` / :func:`context_cost`
/ :func:`file_footprint`), narrowed to the global sidebar / chart-click selection
via :func:`analytics.filter_prompt_ids` so it honours the same filter as every
other tab. Read-only: language is not a global filter dimension, so the charts
emit no cross-filter. Metrics only -- no source code is ever read here.
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
    (0, 50, "0–50"),
    (50, 100, "50–100"),
    (100, 150, "100–150"),
    (150, 200, "150–200"),
    (200, 300, "200–300"),
    (300, 400, "300–400"),
    (400, 500, "400–500"),
    (500, 750, "500–750"),
    (750, 1000, "750–1k"),
    (1000, 1500, "1k–1.5k"),
    (1500, 2000, "1.5k–2k"),
    (2000, 3000, "2k–3k"),
    (3000, 5000, "3k–5k"),
    (5000, 7500, "5k–7.5k"),
    (7500, None, "7.5k+"),
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
    option["grid"] = {"left": 64, "right": 24, "top": 64, "bottom": 72, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["xAxis"] = echarts.category_axis(labels)
    option["xAxis"]["name"] = "Characters"
    option["xAxis"]["nameLocation"] = "middle"
    option["xAxis"]["nameGap"] = 52
    # Many fine-grained bins -> rotate the tick labels so they read as a real
    # distribution without overlapping.
    option["xAxis"]["axisLabel"] = {"color": c["text"], "rotate": 45, "fontSize": 10}
    option["yAxis"] = echarts.value_axis(name="Prompts")
    option["yAxis"]["nameLocation"] = "middle"
    option["yAxis"]["nameGap"] = 46
    option["series"] = [
        {
            "type": "bar",
            "data": counts,
            "itemStyle": {"color": theme.PALETTE[1], "borderRadius": [3, 3, 0, 0]},
            "label": {"show": True, "position": "top", "color": c["text"], "fontSize": 10},
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
    """Cost-by-category breakdown of the input half of the spine (ex-Prompts tab)."""
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
            # Taller than the category bar on the left so the two plot areas bottom out
            # at the same line: this chart reserves more space for its rotated x-axis
            # labels + axis name, which would otherwise float its baseline higher.
            clicked = echarts.render(dist, key="comp_char_dist", height="420px", click=True)
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
# Files section (Axe C+D / DASH5) -- the cost graph: files as centres of gravity
# (real, named, every-user identity), the prompts that edited them orbiting as
# satellites. The most telling, never-random level of the "cost by content" spine
# (the task constellation was retired: inferred task names read as noise).
# ---------------------------------------------------------------------------

# How many file centres the headline graph shows before the long tail is hidden
# (a force layout stays legible at a few dozen nodes, not hundreds).
_FILE_TOP = 40
# File-node radius band (px), mapped from cost by a sqrt scale so area ~ cost.
_FILE_MIN_SIZE, _FILE_MAX_SIZE = 16.0, 56.0
# Edit-satellite radius band (px), much smaller so centres read as centres.
_SAT_MIN_SIZE, _SAT_MAX_SIZE = 6.0, 16.0


def _project_options(prompts: pd.DataFrame) -> list[str]:
    """Projects present in range, busiest first (the file-graph scope choices).

    A constellation mixing several projects is just noise (unrelated trees,
    colliding file names); the page shows **one project at a time**. Order by
    prompt volume so the default lands on the busiest project.
    """
    if "project" not in prompts.columns:
        return []
    values = prompts["project"].dropna().astype(str)
    values = values[values != ""]
    if values.empty:
        return []
    return [str(p) for p in values.value_counts().index.tolist()]


def _cat_label(category: str) -> str:
    """Legend-friendly category label (``(uncategorized)`` stays as-is)."""
    return category if category.startswith("(") else category.title()


def _lang_label(language: str) -> str:
    """Human label for a language (the no-language bucket reads as a dash)."""
    return "—" if language == NO_LANGUAGE else language


def _scaled_size(value: float, vmax: float, lo: float, hi: float) -> float:
    """Map a magnitude to a radius on a sqrt scale (area ~ value), clamped to [lo, hi]."""
    if vmax <= 0:
        return lo
    return float(round(lo + (hi - lo) * (max(value, 0.0) / vmax) ** 0.5, 1))


def _file_graph_option(
    graph: analytics.FileGraph, focus: str | None = None
) -> dict[str, Any] | None:
    """ECharts force-layout graph: file centres (size = context cost, hue = language)
    with the prompts that edited them as satellites; ``focus`` narrows it to one file.

    Colour is driven by ``language`` (the legend), so toggling a language in the
    legend dims every file of that language and its edits at once. Files carry a
    label and a rich tooltip; edit-satellites stay quiet (colour + hover only). A
    file with no satellite is read-only -- pure context rent.
    """
    files = [f for f in graph.files if focus is None or f.path == focus]
    if not files:
        return None
    kept = {f.path for f in files}
    sats = [s for s in graph.satellites if s.path in kept]

    # Stable language -> legend index, in cost order so the legend reads top-down.
    langs: list[str] = []
    for lang in [f.language for f in files]:
        if lang not in langs:
            langs.append(lang)
    lang_index = {lang: i for i, lang in enumerate(langs)}
    color_of = theme.language_color_map(langs)
    lang_of_path = {f.path: f.language for f in files}

    cost_max = max((f.cost for f in files), default=0.0)
    churn_max = max((float(s.churn) for s in sats), default=0.0)
    # Focusing one file zooms in: give its centre and edits more room.
    file_lo, file_hi = (_FILE_MIN_SIZE, _FILE_MAX_SIZE) if focus is None else (40.0, 90.0)
    sat_lo, sat_hi = (_SAT_MIN_SIZE, _SAT_MAX_SIZE) if focus is None else (12.0, 30.0)

    nodes: list[dict[str, Any]] = []
    for f in files:
        short = f.path.rsplit("/", 1)[-1]
        label = short if len(short) <= 26 else short[:25] + "…"
        nodes.append(
            {
                "id": f.path,
                "name": f.path,
                "kind": "file",
                "category": lang_index[f.language],
                "symbolSize": _scaled_size(f.cost, cost_max, file_lo, file_hi),
                "value": round(f.cost, 2),
                "cost": round(f.cost, 2),
                "lang": _lang_label(f.language),
                "fkind": f.kind,
                "edits": f.edits,
                "added": f.lines_added,
                "deleted": f.lines_deleted,
                "reads": f.reads,
                "edited": f.edited,
                "itemStyle": {"color": color_of[f.language], "borderColor": "#0B1220"},
                "label": {"show": True, "formatter": label},
            }
        )
    for s in sats:
        lang = lang_of_path.get(s.path, NO_LANGUAGE)
        nodes.append(
            {
                "id": f"{s.prompt_id}::{s.path}",
                "name": "edit",
                "kind": "edit",
                "category": lang_index.get(lang, 0),
                "symbolSize": _scaled_size(float(s.churn), churn_max, sat_lo, sat_hi),
                "cat": _cat_label(s.category),
                "churn": s.churn,
                "itemStyle": {"color": color_of[lang], "opacity": 0.8},
                "label": {"show": False},
            }
        )
    links = [{"source": s.path, "target": f"{s.prompt_id}::{s.path}"} for s in sats]

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Cost by file — what the work touched",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {
        "data": [_lang_label(lang) for lang in langs],
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
            "if(d.kind==='file'){var ro=d.edited?'':' · read-only';"
            "return '<b>'+d.name+'</b><br/>'+d.lang+' · '+d.fkind+ro+'<br/>$'"
            "+Number(d.cost).toFixed(2)+' context · '+d.edits+' edits (+'+d.added"
            "+'/-'+d.deleted+') · '+d.reads+' reads';}"
            "return 'edit · '+d.cat+' · '+d.churn+' lines';}"
        ),
    }
    option["series"] = [
        {
            "type": "graph",
            "layout": "force",
            "roam": True,
            "draggable": True,
            "categories": [
                {"name": _lang_label(lang), "itemStyle": {"color": color_of[lang]}}
                for lang in langs
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


def _render_files_section(graph: analytics.FileGraph, project: str | None) -> None:
    """The file cost-graph: KPI row, a focus picker, then the force-layout graph."""
    shown = graph.files

    cols = st.columns(4)
    cols[0].metric("Files touched", f"{graph.total_files:,}")
    cols[1].metric("Edited", f"{graph.edited_files:,}", delta="written to", delta_color="off")
    cols[2].metric(
        "Read-only",
        f"{graph.readonly_files:,}",
        delta="pure context rent",
        delta_color="off",
    )
    cols[3].metric(f"Context cost ({graph.provider})", f"${graph.context_total:,.2f}")

    # Drill by file: the headline shows the top centres, the picker zooms one in.
    labels = {f"{f.path[:52]}  ·  ${f.cost:,.2f}": f.path for f in shown}
    choice = st.selectbox(
        "Focus on a file",
        ["All files (top centres)", *labels.keys()],
        key="comp_file_focus",
        label_visibility="collapsed",
    )
    focus = labels.get(choice)

    option = _file_graph_option(graph, focus)
    if option is not None:
        echarts.render(option, key="comp_file_graph", height="560px")

    if focus is None:
        scope = f"**{project}** · " if project else ""
        st.caption(
            f"{scope}{graph.total_files:,} files, **${graph.context_total:,.2f}** in context "
            f"cost (reconciled — the one-off load + the rent each file paid while cached). Each "
            f"**file** is a centre sized by that cost and coloured by language; the **prompts "
            f"that edited it** orbit as satellites (hover for intent + lines). A lone centre "
            f"with no satellite is **read-only** — pure rent, the first thing to keep out of "
            f"context. Top {min(len(shown), _FILE_TOP)} shown — pick one above to zoom in, drag "
            f"nodes, scroll to zoom, toggle a language in the legend."
        )
    else:
        f0 = next(f for f in shown if f.path == focus)
        churn = (
            f"{f0.edits:,} edits (+{f0.lines_added:,}/−{f0.lines_deleted:,})"
            if f0.edited
            else "read-only"
        )
        st.caption(
            f"**{f0.path}** — {_lang_label(f0.language)} · {f0.kind}, **${f0.cost:,.2f}** "
            f"context, {churn}, {f0.reads:,} reads. Each satellite is a prompt that edited it, "
            f"sized by the lines it changed."
        )
        if st.button("Open in File Explorer →", key="comp_open_file_explorer"):
            st.session_state["drill_file"] = focus
            st.session_state["drill_file_project"] = project or ""
            st.switch_page("pages/12_file_explorer.py")


def _render_files(ds: analytics.Dataset, provider: str, prompts: pd.DataFrame) -> None:
    """Files section: scope to a single project, then draw the cost-graph.

    The **project filter** sits above the per-file focus: a constellation mixing
    several project trees is noise, so one project at a time keeps it legible. The
    chosen project narrows the dataset (sessions, edits *and* the session-grained
    context cost via :func:`analytics.filter_project`), so the KPIs, the reconciled
    context cost and the graph all describe that project only.
    """
    project: str | None = None
    scoped = ds
    options = _project_options(prompts)
    if options:
        project = st.selectbox(
            "Project",
            options,
            key="comp_file_project",
            help="Files are scoped per project — the graph shows one project at a time.",
        )
        scoped = analytics.filter_project(ds, project)

    graph = analytics.file_graph(scoped, provider, top=_FILE_TOP)
    if graph.has_data:
        _render_files_section(graph, project)
    else:
        st.info("No file data for this project in the current range.")


# ---------------------------------------------------------------------------
# Page.
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Composition page: input -> output -> context, then Files."""
    st.title("Composition")
    st.caption(
        "Where your cost goes, **by content** — read as one spine: **input** (what you "
        "ask) → **output** (what Claude produces) → **context** (what fills the cache it "
        "re-reads), then **files** (what the work touched). Per-file search + drill lives "
        "on the **File Explorer** page."
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
        "Files — the cost graph",
        "What the work touched: files as centres of gravity (size = context cost, "
        "colour = language), the prompts that edited each one orbiting around it.",
    )
    if not ds.output_files and not ds.context_cost:
        st.info(
            "No file data yet. The per-file footprint (edits crossed with context cost) ships "
            "with the latest extractor — re-run `prompt-analytics extract`, then revisit this "
            "page."
        )
    else:
        _render_files(ds, primary, prompts)


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
