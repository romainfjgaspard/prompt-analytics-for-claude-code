"""Shared ECharts layer for the dashboard.

The dashboard's only chart engine: Apache ECharts via ``streamlit-echarts``
(it replaced the former Plotly render path, see ``docs/MIGRATION-ECHARTS.md``).
Color *semantics* (token types, model families, categories, project colors) and
the typographic helpers stay in ``theme.py`` and are reused here -- this module
only owns the chart engine:

* :func:`colors` / :func:`base_option` -- a theme-aware base (transparent
  background, axes/grid/tooltip colored from the active light/dark theme), so a
  chart follows Streamlit's native appearance switch with no per-page code;
* :func:`render` -- one ``st_echarts`` wrapper; pass ``click=True`` to bind
  ECharts' native ``click`` event and get the clicked category back in Python;
* :func:`apply_click` -- turn that click into a global filter (Power-BI style),
  guarding against the component's sticky value re-applying on every rerun.

Why a real server is required: ``streamlit-echarts`` registers its frontend in
the component registry that the Streamlit *runtime* builds at server start. A
bare import / ``AppTest`` has an empty registry and raises "must be declared in
pyproject.toml"; under ``streamlit run`` it works. Dashboard logic that needs to
stay unit-testable therefore lives in ``data.py`` / ``filters.py``, asserted on
the dataframes that feed these options -- never on the option dicts themselves.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from streamlit_echarts import JsCode, st_echarts

FONT = "Space Grotesk, Inter, sans-serif"


def js(code: str) -> JsCode:
    """Wrap a JS expression so it survives into the option as a real function.

    ``streamlit-echarts`` only turns a string into executable JS when it is a
    :class:`JsCode` (event handlers in the ``events`` dict are wrapped for you;
    a bare function string left inside an *option* -- e.g. a ``label.formatter``
    or ``tooltip.formatter`` -- is serialized as plain text and renders
    literally). Use this for any in-option formatter that needs logic a template
    string (``{b}``/``{c}``/``{d}``/``{value}``) cannot express.
    """
    return JsCode(code)


def label_js(mapping: dict[str, str]) -> JsCode:
    """A formatter mapping a raw value to a display label (else the value itself).

    For ``legend.formatter`` / ``axisLabel.formatter``, which receive the raw
    category string: lets a chart keep raw ids as series/category keys (color,
    cross-filter) while showing friendly labels (e.g. ``theme.model_label``).
    """
    return JsCode("function(v){var M=" + json.dumps(mapping) + ";return M[v]!==undefined?M[v]:v;}")


def model_tooltip_js(mapping: dict[str, str], *, money: bool = False) -> JsCode:
    """An axis-trigger tooltip formatter that shows display labels per series.

    Companion to :func:`label_js` for the multi-series (stacked) tooltip, which
    receives an array of params: each series' raw ``seriesName`` is mapped to its
    display label (else kept) while the bar keeps the raw id as its key.
    """
    fmt = "'$'+Number(p.value).toFixed(2)" if money else "p.value"
    return JsCode(
        "function(ps){var M=" + json.dumps(mapping) + ";"
        "if(!ps||!ps.length){return '';}"
        "var s=ps[0].axisValueLabel+'<br/>';"
        "ps.forEach(function(p){"
        "var n=M[p.seriesName]!==undefined?M[p.seriesName]:p.seriesName;"
        "s+=p.marker+n+': '+(" + fmt + ")+'<br/>';});"
        "return s;}"
    )


def is_dark() -> bool:
    """True when the active Streamlit theme is dark (defaults to dark)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return True


def _theme_key(key: str) -> str:
    """Suffix a component key with the active theme so a light/dark switch remounts.

    ``streamlit-echarts`` reuses a chart instance while its ``key`` is stable, so
    on a theme toggle it keeps the *previous* theme's axis / grid / tooltip colors
    until a hard refresh. Folding the theme into the key gives each theme its own
    instance: switching themes changes the key, Streamlit mounts a fresh chart,
    and the new ``colors()`` apply immediately -- no manual cache clear.

    The separator is a single ``-`` on purpose: the bidirectional-component id
    system reserves the ``__`` delimiter and raises if a key contains it.
    """
    return f"{key}-{'dark' if is_dark() else 'light'}"


def colors() -> dict[str, str]:
    """Theme-aware text / axis / grid / tooltip colors (deep navy / slate-white).

    Mirrors the foundation palette in ``.streamlit/config.toml`` so ECharts
    chrome (axes, gridlines, tooltips, hero sub-text) sits on the same navy
    (dark) / slate-white (light) the rest of the app uses. See the design brief
    (``docs/streamlit_design_system_claude.md``). Data colors are theme-agnostic
    and live in ``theme.py``.
    """
    if is_dark():
        return {
            "text": "#F8FAFC",  # brief Text Primary
            "muted": "#94A3B8",  # brief Text Secondary
            "axis": "#2B3954",  # brief Border
            "grid": "#172033",  # brief Panel/Card (faint gridlines)
            "tooltip_bg": "#172033",
        }
    return {
        "text": "#0F172A",
        "muted": "#64748B",
        "axis": "#CBD5E1",
        "grid": "#E2E8F0",
        "tooltip_bg": "#FFFFFF",
    }


def base_option(**overrides: Any) -> dict[str, Any]:
    """A themed base ECharts option; ``overrides`` shallow-merge on top."""
    c = colors()
    option: dict[str, Any] = {
        "backgroundColor": "transparent",
        "textStyle": {"color": c["text"], "fontFamily": FONT},
        "grid": {"left": 56, "right": 24, "top": 48, "bottom": 56, "containLabel": True},
        "legend": {"bottom": 0, "textStyle": {"color": c["text"]}, "icon": "roundRect"},
        "tooltip": {
            "backgroundColor": c["tooltip_bg"],
            "borderColor": c["axis"],
            "textStyle": {"color": c["text"]},
        },
    }
    option.update(overrides)
    return option


def value_axis(*, money: bool = False, name: str | None = None) -> dict[str, Any]:
    """A themed value axis (optionally ``$`` formatted)."""
    c = colors()
    axis: dict[str, Any] = {
        "type": "value",
        "axisLabel": {"color": c["text"]},
        "axisLine": {"lineStyle": {"color": c["axis"]}},
        "splitLine": {"lineStyle": {"color": c["grid"]}},
    }
    if money:
        axis["axisLabel"]["formatter"] = "${value}"
    if name:
        axis["name"] = name
        axis["nameTextStyle"] = {"color": c["muted"]}
    return axis


def category_axis(data: list[str], *, inverse: bool = False) -> dict[str, Any]:
    """A themed category axis."""
    c = colors()
    return {
        "type": "category",
        "data": data,
        "inverse": inverse,
        "axisLabel": {"color": c["text"]},
        "axisLine": {"lineStyle": {"color": c["axis"]}},
    }


def render(
    option: dict[str, Any],
    *,
    key: str,
    height: str = "380px",
    click: bool = False,
    click_field: str = "name",
) -> Any:
    """Render ``option`` and return the clicked value when ``click`` is set.

    With ``click=True`` an ECharts ``click`` handler returns ``params[click_field]``
    (the category name by default); its value round-trips to Python inside the
    component value under ``chart_event``. Returns that value, or ``None``.
    """
    events = {"click": f"function(p){{ return p.{click_field}; }}"} if click else None
    result = st_echarts(option, events=events, height=height, key=_theme_key(key))
    if isinstance(result, dict):
        return result.get("chart_event")
    return None


def apply_click(
    value: Any,
    filter_key: str,
    *,
    synthetic: frozenset[str] = frozenset(),
) -> None:
    """Apply a single clicked category to a global multiselect filter key.

    Power-BI-style cross-filter: clicking a bar restricts the dashboard to that
    category. Guards against the component's *sticky* value re-applying every
    rerun (only acts when it actually changes the filter), and ignores synthetic
    buckets like ``(session overhead)``. Triggers a rerun on a real change.
    """
    if not isinstance(value, str) or value in synthetic:
        return
    if st.session_state.get(filter_key) == [value]:
        return
    # The filter key is a sidebar multiselect's widget key, which cannot be set
    # directly after that widget is instantiated this run. Stage it; the sidebar
    # applies it before its widgets render next run. (The guard above stops the
    # component's sticky value from re-staging the same pick on every rerun.)
    from prompt_analytics.dashboard import filters

    filters.stage_filter(filter_key, [value])
    st.rerun()


def render_events(
    option: dict[str, Any],
    *,
    key: str,
    height: str = "380px",
    events: dict[str, str],
) -> dict[str, Any]:
    """Render with arbitrary ECharts event handlers; return the raw value dict.

    The single chart can wire several handlers (e.g. ``click`` + ``brushEnd``);
    each handler's return value round-trips into the **same** ``chart_event``
    field, so the caller disambiguates by *type* (a clicked category is a
    ``str``; a brushed range is a ``list``). Returns ``{}`` when nothing fired.
    """
    result = st_echarts(option, events=events, height=height, key=_theme_key(key))
    return result if isinstance(result, dict) else {}


def brush_toolbox(option: dict[str, Any]) -> None:
    """Add an x-only (``lineX``) date brush + its toolbox toggle, in place.

    The brush is bound to ``xAxisIndex: 0`` so ECharts fills ``area.coordRange``
    with the *category-index* range of the selection -- which :func:`date_brush_js`
    turns back into date labels. A toolbox button toggles brush mode on; until
    then the chart's ``click`` handler stays live, so a plain click on a bar
    still emits a single-day filter (the proven fallback path).
    """
    c = colors()
    option["brush"] = {
        "xAxisIndex": 0,
        "brushType": "lineX",
        "brushMode": "single",
        "throttleType": "debounce",
        "throttleDelay": 200,
        "brushStyle": {"borderColor": "#D97757", "color": "rgba(217,119,87,0.15)"},
    }
    toolbox = option.get("toolbox", {})
    toolbox.setdefault("show", True)
    toolbox.setdefault("right", 16)
    toolbox["feature"] = {**toolbox.get("feature", {}), "brush": {"type": ["lineX", "clear"]}}
    toolbox["iconStyle"] = {"borderColor": c["text"]}
    option["toolbox"] = toolbox


def date_brush_js(labels: list[str]) -> str:
    """JS ``brushEnd`` handler mapping an x-axis brush to ``[startLabel, endLabel]``.

    Reads ``area.coordRange`` (the category-index range ECharts computes for a
    ``lineX`` brush bound to ``xAxisIndex: 0``), rounds to the nearest bars and
    returns the two date labels embedded from ``labels``. Returns ``undefined``
    for an empty / cleared brush, which the component treats as client-side only
    (no round-trip, no rerun) -- so clearing the brush does not fire a filter.
    """
    arr = json.dumps(labels)
    return (
        "function(params){"
        "var L=" + arr + ";"
        "if(!params.areas||!params.areas.length){return;}"
        "var cr=params.areas[params.areas.length-1].coordRange;"
        "if(!cr||cr.length<2){return;}"
        "var a=Math.round(cr[0]),b=Math.round(cr[1]);"
        "if(a>b){var t=a;a=b;b=t;}"
        "if(a<0){a=0;}if(b>L.length-1){b=L.length-1;}"
        "if(b<0||a>L.length-1){return;}"
        "return [L[a],L[b]];"
        "}"
    )


def _norm_range(value: Any) -> tuple[str, str] | None:
    """Normalize a stored date-range to a ``(isoStart, isoEnd)`` string pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (str(value[0])[:10], str(value[-1])[:10])


def apply_date_range(value: Any, filter_key: str) -> None:
    """Apply a clicked day (``str``) or brushed range (``list``) to a date filter.

    Sister of :func:`apply_click` for the date dimension. Guards against the
    component's *sticky* value re-applying every rerun by comparing **normalized
    ISO strings**: the sidebar rewrites this key as ``datetime.date`` objects, so
    a naive ``==`` between dates and the brush's string labels would never match
    and would loop forever. Triggers a rerun only on a real change.
    """
    if isinstance(value, str):
        start = end = value
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = str(value[0]), str(value[1])
    else:
        return
    if _norm_range(st.session_state.get(filter_key)) == (start[:10], end[:10]):
        return
    # ``filter_key`` is the sidebar date_input's widget key, which cannot be set
    # directly after the widget is instantiated this run; stage it like the
    # categorical cross-filters (the sidebar normalizes the strings to dates).
    from prompt_analytics.dashboard import filters

    filters.stage_filter(filter_key, [start, end])
    st.rerun()


def sparkline_option(values: list[float], color: str) -> dict[str, Any]:
    """A tiny axis-less area line for a KPI (the ECharts sparkline, 7.2)."""
    from prompt_analytics.dashboard import theme

    return {
        "backgroundColor": "transparent",
        "grid": {"left": 2, "right": 2, "top": 2, "bottom": 2},
        "xAxis": {"type": "category", "show": False, "data": list(range(len(values)))},
        "yAxis": {"type": "value", "show": False},
        "tooltip": {"show": False},
        "series": [
            {
                "type": "line",
                "data": values,
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {"color": color, "width": 1.5},
                "areaStyle": {"color": theme._rgba(color, 0.15)},
            }
        ],
    }
