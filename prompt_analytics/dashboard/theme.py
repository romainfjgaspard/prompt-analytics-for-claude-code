"""Shared visual identity for every dashboard chart.

One palette, shared across pages so the charts look consistent wherever the app
runs. The rendering itself lives in :mod:`prompt_analytics.dashboard.echarts`
(Apache ECharts); this module only owns the **color semantics**, the typographic
hierarchy (:func:`section`, :func:`hero`) and small text helpers. The old Plotly
template / render path was removed once the whole board moved to ECharts
(``docs/MIGRATION-ECHARTS.md``).

Color tells one story: **tokens describe the technical mechanics of cost; models
describe the Claude product range.** The two axes are split not by hand-picked
hues that merely avoid collisions, but by *signature* -- a model chart is a
single-hue coral ramp, a token chart is a cool multicolor set -- so a glance at
any chart on any page says which axis you are reading, with no legend relearning.

* a **UI palette** (neutral chrome: background, panels, text) -- in
  ``.streamlit/config.toml`` and :func:`echarts.colors`, not here;
* a **Token palette** (:data:`TOKEN_TYPE_COLORS` + :data:`SEMANTIC_COLORS`) --
  token charts only, a cool "flow" set: input the blue, output the cyan (what
  goes in / what comes out), cache read the green (retrieval), cache writes the
  purples (writing); errors/warnings/subagents the semantic accents. No warm
  hue, so nothing competes with the Claude coral;
* a **Model palette** (:data:`_FAMILY_SHADES`) -- model charts only, shades of
  the **Anthropic clay/coral**: lightest for the smallest model, darkest for the
  largest, the newest Opus on the official Claude color. The whole lineup reads
  as one family.

Mappings are constant across every page and identical in both themes (only the
UI chrome flips light/dark). Prompt categories (:data:`CATEGORY_COLORS`) are a
secondary axis confined to the Prompts page, grouped by intent (critical ops /
reflection / support).
* **Projects** get a stable hue (:func:`project_color_map`) so a project keeps
  the same color across the pareto, treemap and scatter, and even when a filter
  changes which other projects are present -- provided the caller builds the map
  once from the full (unfiltered) project universe and shares it (see the
  function's docstring).
* **Categories** keep one hue per label (debug = red, refactor = purple, ...).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import streamlit as st

__all__ = [
    "PALETTE",
    "TOKEN_TYPE_COLORS",
    "SEMANTIC_COLORS",
    "CATEGORY_COLORS",
    "LANGUAGE_COLORS",
    "language_color_map",
    "MIX_COLORS",
    "model_color_map",
    "model_label",
    "model_sort_key",
    "sort_models",
    "project_color_map",
    "section",
    "hero",
    "md_escape",
]


def md_escape(text: str) -> str:
    """Escape ``$`` for Streamlit markdown widgets (captions, alerts).

    Analytics notes carry dollar amounts; two ``$`` in one string would
    otherwise open a LaTeX math span and silently mangle the sentence.
    """
    return text.replace("$", "\\$")


# Qualitative default palette (colorblind-aware, anchored on the brand coral).
PALETTE = [
    "#D97757",  # coral
    "#3B82F6",  # blue
    "#10B981",  # green
    "#8B5CF6",  # purple
    "#F59E0B",  # amber
    "#14B8A6",  # teal
    "#EC4899",  # pink
    "#6B7280",  # gray
]

# Keyed by display label (token_type_label) AND machine key, so both work.
# The Token palette: a cool "flow" set (no warm hue, so nothing competes with the
# Claude coral the models own). Keyed by display label AND machine key.
TOKEN_TYPE_COLORS = {
    "Input": "#3B82F6",  # bleu — ce qui entre
    "Output": "#06B6D4",  # cyan — ce qui sort
    "Cache read": "#10B981",  # vert — récupération
    "Cache write (5m)": "#A78BFA",  # violet clair — écriture
    "Cache write (1h)": "#8B5CF6",  # violet — écriture
    "Server tool use": "#9CA3AF",
    "input": "#3B82F6",
    "output": "#06B6D4",
    "cache_read": "#10B981",
    "cache_write_5m": "#A78BFA",
    "cache_write_1h": "#8B5CF6",
    "server_tool_use": "#9CA3AF",
}

# Semantic accents that live alongside the token palette (errors / warnings /
# subagents). Distinct but calm; reused for status, never for a model.
SEMANTIC_COLORS = {
    "errors": "#EF4444",
    "warnings": "#F59E0B",
    "subagents": "#14B8A6",
}

# Token-mix components (Session depth page). The four input-side components --
# cache writes stay split by TTL (1.3): 1h writes are billed 2x and can dominate
# the 5m bucket 10:1 -- plus **Output**, shown as a fifth panel beside them so the
# page reads what a prompt produces, not only what it consumes. Output keeps its
# token-palette cyan (matches ``TOKEN_TYPE_COLORS["Output"]``).
MIX_COLORS = {
    "Fresh input": "#3B82F6",
    "Cache read": "#10B981",
    "Cache write (5m)": "#A78BFA",
    "Cache write (1h)": "#8B5CF6",
    "Output": "#06B6D4",
}

# Prompt categories (Prompts page only): grouped by intent so the legend keeps a
# mental logic instead of reading as a rainbow -- critical ops, reflection work,
# support. Confined to one page, so echoing a model/token hue here is harmless.
CATEGORY_COLORS = {
    # Critical ops
    "debug": "#DC2626",
    "ops": "#2563EB",
    "implementation": "#0EA5A4",
    "refactor": "#7C3AED",
    # Reflection work
    "question": "#D97706",
    "plan": "#6366F1",
    "review": "#0891B2",
    "followup": "#64748B",
    # feedback = course-correction/steering: a slate kin to followup but warmer,
    # so the two conversational buckets read as a family yet stay distinct.
    "feedback": "#9F7AEA",
    # Support
    "docs": "#BE185D",
    "test": "#65A30D",
    # notification = harness plumbing (background-task finished), not a prompt;
    # a muted grey near "other" since it's normally hidden from the category view.
    "notification": "#CBD5E1",
    "other": "#94A3B8",
    "(uncategorized)": "#D1D5DB",
}

# The Model palette: every model is a shade of the Anthropic clay/coral, so a
# model chart reads instantly as "the Claude lineup" and never collides with the
# (cool, coral-free) token palette. Bands are ordered by weight around the
# official Claude color (#CC785C = Opus): haiku much lighter, sonnet lighter,
# opus *is* the Claude color, fable much darker (mythos darker still). Within a
# family, versions step to neighboring tints of the same band so a family stays
# one recognizable group. The first (base) shade is the family's representative
# color; the band spacing carries the hierarchy, the legend disambiguates versions.
_FAMILY_SHADES: dict[str, list[str]] = {
    "haiku": ["#F2C9B9", "#F8DCD0", "#ECBBA8", "#FBEAE2"],  # much lighter
    "sonnet": ["#E1A084", "#EAB39B", "#D78D6C", "#F0C6B2"],  # lighter
    "opus": ["#CC785C", "#D2826A", "#C36C50", "#D98E74"],  # the official Claude color
    "fable": ["#9E4F2D", "#B36241", "#8A4225", "#C5775A"],  # much darker
    "mythos": ["#6F3520", "#834428", "#5C2A18", "#985C42"],  # darkest, rare
}
# Non-Claude models (no family match) fall back to neutral grey -- they are not
# part of the Claude lineup, so they should not borrow its coral.
_FALLBACK_SHADES = ["#6B7280", "#9CA3AF", "#4B5563", "#D1D5DB"]

# Distinct hues for projects (hash-indexed). Wider than PALETTE so collisions
# between real project names stay unlikely.
_PROJECT_PALETTE = [
    "#D97757",  # coral
    "#3B82F6",  # blue
    "#10B981",  # green
    "#8B5CF6",  # purple
    "#F59E0B",  # amber
    "#14B8A6",  # teal
    "#EC4899",  # pink
    "#6366F1",  # indigo
    "#84CC16",  # lime
    "#06B6D4",  # cyan
    "#F43F5E",  # rose
    "#A855F7",  # violet
    "#0EA5E9",  # sky
    "#EAB308",  # yellow
]
# Synthetic "non-project" buckets always read as neutral grey, never a hue.
_PROJECT_SPECIAL = {
    "(session overhead)": "#9CA3AF",
    "(unknown)": "#D1D5DB",
}


# Families ordered smallest -> largest; the order stacked charts and legends
# should follow, mirroring the light -> dark coral ramp in _FAMILY_SHADES.
_FAMILY_ORDER = ["haiku", "sonnet", "opus", "fable", "mythos"]


def model_sort_key(model: str) -> tuple[int, str]:
    """Sort key ordering models smallest-family-first, then by version.

    Stacked charts and legends should read Haiku -> Sonnet -> Opus -> Fable (the
    same light -> dark order as the coral ramp), never alphabetically. Non-Claude
    models (no family match) sort last, then alphabetically.
    """
    lower = model.lower()
    for rank, family in enumerate(_FAMILY_ORDER):
        if family in lower:
            return (rank, model)
    return (len(_FAMILY_ORDER), model)


def sort_models(models: Iterable[str]) -> list[str]:
    """Models ordered smallest family first (see :func:`model_sort_key`)."""
    return sorted(set(models), key=model_sort_key)


_FAMILY_LABELS = {f: f.capitalize() for f in _FAMILY_ORDER}
_DATE_STAMP = re.compile(r"^\d{6,8}$")


def model_label(model: str) -> str:
    """Friendly **display** name for a model id (legends/axes/tooltips only).

    Drops the ``claude-`` vendor prefix and the trailing date stamp, and turns
    the version digits into a dotted number::

        claude-haiku-4-5-20251001  ->  Haiku 4.5
        claude-opus-4-8            ->  Opus 4.8
        claude-fable-5             ->  Fable 5

    The raw id stays the data key (color, sort, cross-filter); only the shown
    text changes. Unknown ids are still cleaned (prefix/date removed, hyphens to
    spaces, title-cased) so nothing ever renders the raw ``claude-...`` string.
    """
    raw = str(model)
    if raw.strip() == "" or raw.strip().lower() == "nan":
        return "(unknown)"
    stripped = re.sub(r"(?i)^claude-", "", raw)
    tokens = [t for t in stripped.split("-") if t and not _DATE_STAMP.match(t)]
    if not tokens:
        return raw
    low = raw.lower()
    family = next((f for f in _FAMILY_ORDER if f in low), "")
    if family:
        version = ".".join(t for t in tokens if t.isdigit())
        return f"{_FAMILY_LABELS[family]} {version}".strip()
    return " ".join(tokens).replace("_", " ").title()


def model_color_map(models: Iterable[str]) -> dict[str, str]:
    """Stable color per model id, grouped by family (opus/sonnet/haiku/...).

    Iterated in family order (:func:`sort_models`) so the per-version shade
    assignment within a family is stable and does not depend on row order.
    """
    counters: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for model in sort_models(models):
        family = next((f for f in _FAMILY_SHADES if f in model.lower()), "")
        shades = _FAMILY_SHADES.get(family, _FALLBACK_SHADES)
        index = counters.get(family, 0)
        mapping[model] = shades[index % len(shades)]
        counters[family] = index + 1
    return mapping


def project_color_map(projects: Iterable[str]) -> dict[str, str]:
    """Stable, **distinct** color per project (analogous to ``model_color_map``).

    Real project names are assigned the brand palette in sorted order (so two
    projects never share a hue while there are colors left, unlike a name-hash
    which collides for small N); the synthetic ``(session overhead)`` /
    ``(unknown)`` buckets are always neutral grey and are present in the result
    even when absent from ``projects``.

    Because the assignment is by sorted position, **stability across charts and
    filters comes from passing the same project universe every time**: build the
    map once from the full, unfiltered project list and share the single dict
    with the pareto, treemap and scatter (Plotly ignores keys a given chart does
    not use). That is what the Sessions page does -- a per-chart map built from
    each chart's own (differently filtered) subset is exactly the instability
    the audit flagged.
    """
    mapping: dict[str, str] = dict(_PROJECT_SPECIAL)
    reals = [p for p in sorted({str(x) for x in projects}) if p not in _PROJECT_SPECIAL]
    for index, project in enumerate(reals):
        mapping[project] = _PROJECT_PALETTE[index % len(_PROJECT_PALETTE)]
    return mapping


# Languages (Composition page only): a confined secondary axis, like categories
# and projects. Recognizable linguist-ish hues for the common languages so a
# reader spots Python / TypeScript at a glance; the synthetic buckets read as
# neutral grey, and any other language gets a stable palette hue by sorted
# position (see :func:`language_color_map`). Confined to one page, so echoing a
# token / model hue here is harmless.
LANGUAGE_COLORS = {
    "Python": "#4B8BBE",
    "TypeScript": "#3178C6",
    "JavaScript": "#E8B339",
    "Jupyter Notebook": "#DA5B0B",
    "Markdown": "#8B949E",
    "Go": "#00ADD8",
    "Rust": "#CE7B53",
    "Java": "#B07219",
    "Kotlin": "#A97BFF",
    "Ruby": "#CC342D",
    "PHP": "#777BB4",
    "C": "#6E7781",
    "C++": "#F34B7D",
    "C#": "#178600",
    "Swift": "#F05138",
    "Scala": "#C22D40",
    "Shell": "#89E051",
    "PowerShell": "#5391FE",
    "SQL": "#3A7CA5",
    "HTML": "#E34C26",
    "CSS": "#663399",
    "Vue": "#41B883",
    "Svelte": "#FF3E00",
    "Dart": "#00B4AB",
    "Elixir": "#6E4A7E",
    "Lua": "#000080",
    "R": "#198CE7",
    "Terraform": "#7B42BC",
    "Dockerfile": "#2496ED",
    "Makefile": "#427819",
    "YAML": "#9CA3AF",
    "JSON": "#9CA3AF",
    "TOML": "#9CA3AF",
    "XML": "#9CA3AF",
    "INI": "#9CA3AF",
    "Text": "#94A3B8",
    "CSV": "#94A3B8",
    "reStructuredText": "#8B949E",
}
# Synthetic non-language buckets always read as neutral grey, never a hue.
_LANGUAGE_SPECIAL = {
    "(other tooling)": "#64748B",
    "(unknown)": "#D1D5DB",
    "other": "#94A3B8",
}


def language_color_map(languages: Iterable[str]) -> dict[str, str]:
    """Stable color per language label (Composition page, analogous to projects).

    Known languages keep their curated linguist-ish hue (:data:`LANGUAGE_COLORS`);
    the synthetic ``(other tooling)`` / ``(unknown)`` buckets are always neutral
    grey; anything else is assigned the brand palette in sorted order so two
    unmapped languages do not collide while colors remain. As with
    :func:`project_color_map`, pass the full language universe so a language
    keeps one hue across the page's charts.
    """
    mapping: dict[str, str] = dict(_LANGUAGE_SPECIAL)
    rest: list[str] = []
    for lang in sorted({str(x) for x in languages}):
        if lang in LANGUAGE_COLORS:
            mapping[lang] = LANGUAGE_COLORS[lang]
        elif lang not in mapping:
            rest.append(lang)
    for index, lang in enumerate(rest):
        mapping[lang] = _PROJECT_PALETTE[index % len(_PROJECT_PALETTE)]
    return mapping


# ---------------------------------------------------------------------------
# Typographic hierarchy (7.5) + hero numbers (7.1).
# ---------------------------------------------------------------------------


def section(title: str, caption: str | None = None) -> None:
    """A named page section: the second title size of the hierarchy (7.5).

    Pages read as named arguments ("Where the money goes" / "When" / ...)
    instead of a flat stack of same-sized chart titles.
    """
    st.header(title, divider="gray")
    if caption:
        st.caption(caption)


def hero(value: str, lead: str, sub: str = "", color: str = PALETTE[0]) -> str:
    """HTML for one hero number (7.1): a very large figure with its claim.

    Returned (not rendered) so the caller can place it in a column; render
    with ``st.markdown(..., unsafe_allow_html=True)``. The lead/sub text colors
    are pulled from the active theme (``echarts.colors``) so the headline stays
    legible on the dark stockpeers slate as well as the light theme -- the old
    hard-coded greys (``#374151``/``#6B7280``) washed out to near-invisible on
    the dark background.
    """
    from prompt_analytics.dashboard import echarts

    c = echarts.colors()
    sub_html = (
        f'<div style="font-size:0.9rem;color:{c["muted"]};line-height:1.35;'
        f'max-width:34rem;">{sub}</div>'
        if sub
        else ""
    )
    return (
        f'<div style="font-size:1.05rem;color:{c["text"]};font-weight:600;">{lead}</div>'
        f'<div style="font-size:4rem;font-weight:750;color:{color};'
        f'line-height:1.05;margin:0.1rem 0 0.3rem 0;">{value}</div>'
        f"{sub_html}"
    )


def _rgba(hex_color: str, alpha: float) -> str:
    """``#RRGGBB`` -> ``rgba(r,g,b,a)`` (for translucent ECharts fills/markers)."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"
