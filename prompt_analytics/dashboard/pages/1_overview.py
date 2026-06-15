"""Overview dashboard page: cost breakdown and spend rhythm.

The pilot page of the Plotly -> ECharts migration (``docs/MIGRATION-ECHARTS.md``).
It validates the **date-range cross-filter** here first, on the cost-by-token-type
trend, because it is the riskiest mechanism and every other page reuses it:

* primary -- a ``lineX`` **brush** over the trend emits ``[startDate, endDate]``
  into ``filters.KEY_DATE_RANGE`` (ECharts ``brushEnd`` -> ``coordRange`` ->
  date labels, see ``echarts.date_brush_js`` / ``apply_date_range``);
* fallback -- a plain **click** on a bar filters to that single day/week, reusing
  the click path already proven on the home page and the spike.

Both share ``KEY_DATE_RANGE`` and are disambiguated by type (str = click, list =
brush). Per-session / per-prompt detail tables moved to the Explorer page (no
tables on the analytical pages); apply a filter here, then use the *Explore →*
button in the filter badge. Everything is ECharts. Zero Plotly remains.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics.dashboard import data, echarts, filters, theme

# Abbreviate large counts on an axis / tooltip (1_500_000 -> "1.5M", 12_000 -> "12k").
_ABBREV_JS = (
    "function(v){v=Number(v);"
    "if(Math.abs(v)>=1e6)return (v/1e6).toFixed(1)+'M';"
    "if(Math.abs(v)>=1e3)return (v/1e3).toFixed(0)+'k';"
    "return ''+v;}"
)


def _cost_by_token_type_option(
    tokens: pd.DataFrame, primary: str
) -> tuple[dict[str, Any] | None, list[str], bool]:
    """Stacked daily (or weekly) bar of cost by token type, + a date brush.

    Aggregates to weekly when the date span exceeds 30 days, which avoids ~160
    hairline bars that read as texture rather than data (6.7). Returns the
    ECharts option, the ordered x-axis date labels (for the brush handler), and
    whether weekly aggregation is active (so a selection can be widened to cover
    the whole week).
    """
    col = data.cost_col(primary)
    daily = (
        tokens.dropna(subset=["date"])[["date", "token_type_label", col]]
        .groupby(["date", "token_type_label"], as_index=False)
        .sum()
        .sort_values("date")
    )
    if daily.empty:
        return None, [], False

    dates = pd.to_datetime(daily["date"], utc=True)
    date_span = int((dates.max() - dates.min()).total_seconds() / 86400) if len(dates) > 1 else 0
    weekly = date_span > 30

    if weekly:
        # Floor each date to the Monday of its ISO week.
        dow = dates.dt.dayofweek  # Monday = 0
        daily = daily.assign(_x=(dates - pd.to_timedelta(dow, unit="D")).dt.strftime("%Y-%m-%d"))
        grain = "week"
    else:
        daily = daily.assign(_x=pd.to_datetime(daily["date"], utc=True).dt.strftime("%Y-%m-%d"))
        grain = "day"

    daily = daily.groupby(["_x", "token_type_label"], as_index=False)[[col]].sum()
    labels = sorted(daily["_x"].unique().tolist())
    pivot = daily.pivot_table(
        index="_x", columns="token_type_label", values=col, aggfunc="sum", fill_value=0.0
    ).reindex(labels)
    series = [
        {
            "name": token_type,
            "type": "bar",
            "stack": "cost",
            "data": [round(float(v), 2) for v in pivot[token_type].tolist()],
            "itemStyle": {"color": theme.TOKEN_TYPE_COLORS.get(token_type, "#9CA3AF")},
        }
        for token_type in pivot.columns
    ]

    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": f"Cost by token type per {grain} ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 56, "right": 24, "top": 48, "bottom": 72, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    option["xAxis"] = echarts.category_axis(labels)
    option["yAxis"] = echarts.value_axis(money=True)
    option["series"] = series
    echarts.brush_toolbox(option)
    return option, labels, weekly


def _expand_weekly(value: Any) -> Any:
    """Widen a week-Monday label to its inclusive 7-day span (``+6`` days).

    The trend's x labels are week-start Mondays when aggregated weekly; a click
    or brush returns those Mondays, so without this the inclusive date filter
    would drop the last six days of the selected week.
    """

    def plus6(d: str) -> str:
        return str((pd.to_datetime(d) + pd.Timedelta(days=6)).strftime("%Y-%m-%d"))

    if isinstance(value, str):
        return [value, plus6(value)]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [str(value[0]), plus6(str(value[1]))]
    return value


def _render_token_volume_section(tokens: pd.DataFrame) -> None:
    """Headline cache-read share as a metric + horizontal bars for the rest (V3).

    Replaces the donut chart: a single dominant slice (cache reads ~= 90 %+)
    makes the donut unreadable; the percentage as a headline number answers the
    real question directly, and a bar of the remaining types shows the detail.
    """
    label = "token_type_label" if "token_type_label" in tokens.columns else "token_type"
    work = tokens[tokens["token_type"] != "server_tool_use"]
    by_type = work.groupby(label, as_index=False)["token_count"].sum()
    total = float(by_type["token_count"].sum())
    if total == 0:
        st.info("No token volume data.")
        return

    cache_vol = float(by_type.loc[by_type[label] == "Cache read", "token_count"].sum())
    cache_pct = 100.0 * cache_vol / total
    st.metric("Cache reads", f"{cache_pct:.1f}% of token volume")

    rest = by_type[by_type[label] != "Cache read"].sort_values("token_count", ascending=True)
    if rest.empty:
        return
    names = rest[label].tolist()
    option = echarts.base_option()
    option["title"] = {
        "text": "Non-cache token volume",
        "left": 0,
        "textStyle": {"color": echarts.colors()["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"show": False}
    option["grid"] = {"left": 8, "right": 40, "top": 48, "bottom": 24, "containLabel": True}
    # Abbreviate the axis (M / k) so a "Tokens" name at the right edge (which
    # clipped) is unnecessary and large counts stay readable.
    option["tooltip"].update(
        {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "valueFormatter": echarts.js(_ABBREV_JS),
        }
    )
    xaxis = echarts.value_axis()
    xaxis["axisLabel"]["formatter"] = echarts.js(_ABBREV_JS)
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.category_axis(names, inverse=True)
    option["series"] = [
        {
            "type": "bar",
            "data": [
                {
                    "value": int(v),
                    "itemStyle": {"color": theme.TOKEN_TYPE_COLORS.get(n, "#9CA3AF")},
                }
                for n, v in zip(names, rest["token_count"].tolist(), strict=True)
            ],
            "itemStyle": {"borderRadius": [0, 4, 4, 0]},
        }
    ]
    echarts.render(option, key="overview_token_volume", height="220px")


def _render_subagents_section(tokens: pd.DataFrame, primary: str) -> None:
    """Subagent (sidechain) cost: KPI + a by-token-type composition bar (7.4, 1.2).

    Sidechain rows are first-class in ``tokens.csv`` (``is_sidechain``); their
    cost is also included in the parent prompt's totals everywhere else, so this
    is a lens, not a new bucket. The breakdown is by **token type** (not model --
    model is the Models page's dimension, out of place here): it stays in this
    page's context-rent narrative and shows that subagent spend is mostly cache
    too. Colors reuse ``theme.TOKEN_TYPE_COLORS`` like the rest of the page.
    """
    col = data.cost_col(primary)
    if "is_sidechain" not in tokens.columns or col not in tokens.columns:
        return
    side_mask = pd.to_numeric(tokens["is_sidechain"], errors="coerce").fillna(0) == 1
    total = float(tokens[col].sum())
    side = tokens[side_mask]
    side_cost = float(side[col].sum())
    if total <= 0:
        return
    st.metric(
        "Subagent cost",
        f"${side_cost:,.2f}",
        delta=f"{100 * side_cost / total:.1f}% of the bill",
        delta_color="off",
    )
    st.caption("Cost of sub-agents launched by your main agents (already in the totals above).")

    label = "token_type_label" if "token_type_label" in side.columns else "token_type"
    if side_cost <= 0 or label not in side.columns:
        return
    by_type = side.groupby(label, as_index=False)[col].sum()
    by_type = by_type[by_type[col] > 0].sort_values(col, ascending=True)
    if by_type.empty:
        return
    names = by_type[label].tolist()
    # One bar per token type (same shape as the "Non-cache token volume" bar above),
    # so no legend is needed -- the y-axis names the types and nothing overlaps.
    option = echarts.base_option()
    option["title"] = {
        "text": "Subagent cost by token type",
        "left": 0,
        "textStyle": {"color": echarts.colors()["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["legend"] = {"show": False}
    option["grid"] = {"left": 8, "right": 40, "top": 48, "bottom": 16, "containLabel": True}
    option["tooltip"].update(
        {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "valueFormatter": echarts.js("function(v){return '$'+Number(v).toFixed(2);}"),
        }
    )
    option["xAxis"] = echarts.value_axis(money=True)
    option["yAxis"] = echarts.category_axis(names, inverse=True)
    option["series"] = [
        {
            "type": "bar",
            "data": [
                {
                    "value": round(float(v), 2),
                    "itemStyle": {"color": theme.TOKEN_TYPE_COLORS.get(str(n), "#9CA3AF")},
                }
                for n, v in zip(names, by_type[col].tolist(), strict=True)
            ],
            "itemStyle": {"borderRadius": [0, 4, 4, 0]},
        }
    ]
    echarts.render(option, key="overview_subagents", height="220px")


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _spend_heatmap_option(
    tokens: pd.DataFrame, primary: str, tz_offset: int = 0
) -> dict[str, Any] | None:
    """Punchcard heatmap: when the money is spent (weekday x hour).

    ``tz_offset`` shifts the UTC timestamps so the heatmap reflects local working
    hours (e.g. ``tz_offset=2`` for UTC+2 / Paris).
    """
    col = data.cost_col(primary)
    if "timestamp" not in tokens.columns:
        return None
    work = tokens.dropna(subset=["timestamp"]).copy()
    if work.empty:
        return None
    ts = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    if tz_offset:
        ts = ts + pd.Timedelta(hours=tz_offset)
    work = work.assign(hour=ts.dt.hour, weekday=ts.dt.day_name())
    grid = work.groupby(["weekday", "hour"], as_index=False)[col].sum()

    hours = [f"{h:02d}h" for h in range(24)]
    wd_index = {name: i for i, name in enumerate(_WEEKDAYS)}
    points = [
        [int(row.hour), wd_index[row.weekday], round(float(getattr(row, col)), 2)]
        for row in grid.itertuples(index=False)
        if row.weekday in wd_index
    ]
    if not points:
        return None
    max_cost = max(p[2] for p in points) or 1.0

    c = echarts.colors()
    dark = echarts.is_dark()
    # Sequential blue ramp (design brief, "Heatmaps"): a cool single-hue scale
    # reads as data intensity and avoids competing with the coral chrome. Dark
    # climbs from the navy background up through Tailwind blues; light mirrors it
    # from a near-white blue up to a deep blue.
    ramp = (
        ["#0B1220", "#1E293B", "#3B82F6", "#60A5FA", "#BFDBFE"]
        if dark
        else ["#F8FAFC", "#BFDBFE", "#60A5FA", "#3B82F6", "#1E40AF"]
    )
    tz_label = f"UTC{tz_offset:+d}" if tz_offset != 0 else "UTC"
    option = echarts.base_option()
    option["title"] = {
        "text": f"When the money is spent ({primary}, {tz_label})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 60, "right": 24, "top": 48, "bottom": 70, "containLabel": True}
    option["tooltip"] = {
        "backgroundColor": c["tooltip_bg"],
        "borderColor": c["axis"],
        "textStyle": {"color": c["text"]},
        "position": "top",
        # Must be wrapped in echarts.js: a bare function string in an option is
        # serialized as text and renders literally (it showed the raw object).
        "formatter": echarts.js(
            "function(p){var H=" + _js_array(hours) + ",D=" + _js_array(_WEEKDAYS) + ";"
            "return D[p.data[1]]+' '+H[p.data[0]]+': $'+Number(p.data[2]).toFixed(2);}"
        ),
    }
    xaxis = echarts.category_axis(hours)
    xaxis["splitArea"] = {"show": True}
    yaxis = echarts.category_axis(_WEEKDAYS, inverse=True)
    yaxis["splitArea"] = {"show": True}
    option["xAxis"] = xaxis
    option["yAxis"] = yaxis
    option["visualMap"] = {
        "min": 0,
        "max": max_cost,
        "calculable": True,
        "orient": "horizontal",
        "left": "center",
        "bottom": 0,
        "textStyle": {"color": c["text"]},
        "inRange": {"color": ramp},
    }
    option["series"] = [
        {
            "type": "heatmap",
            "data": points,
            "emphasis": {"itemStyle": {"shadowBlur": 8, "shadowColor": "rgba(0,0,0,0.4)"}},
        }
    ]
    return option


def _js_array(values: list[str]) -> str:
    """Embed a list of strings as a JS array literal for an in-handler lookup."""
    return json.dumps(values)


def main() -> None:
    """Render the Overview page."""
    st.title("Overview")

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all)
    tokens = frames.get("tokens", pd.DataFrame())
    primary = data.primary_provider()

    if tokens.empty or data.cost_col(primary) not in tokens.columns:
        st.info("No data for the current filters.")
        st.stop()

    # Headline number for this page: how much of the API-equivalent cost is spent
    # re-sending context rather than generating answers. The cost-by-token-type
    # trend right below is its direct support.
    rent = data._context_rent_share(tokens, primary)
    if rent is not None:
        st.markdown(
            theme.hero(
                f"{rent:.0f}%",
                "of your API-equivalent cost is context rent",
                "cache reads + cache writes: money spent re-sending context "
                "at every turn, not generating answers.",
            ),
            unsafe_allow_html=True,
        )

    theme.section("Where the money goes")
    left, right = st.columns([3, 2])
    with left:
        option, labels, weekly = _cost_by_token_type_option(tokens, primary)
        if option is None:
            st.info("No dated cost for the current filters.")
        else:
            result = echarts.render_events(
                option,
                key="overview_cost_by_type",
                height="620px",
                events={
                    "click": "function(p){ return p.name; }",
                    "brushEnd": echarts.date_brush_js(labels),
                },
            )
            value = result.get("chart_event")
            if value is not None and weekly:
                value = _expand_weekly(value)
            echarts.apply_date_range(value, filters.KEY_DATE_RANGE)
            st.caption(
                "👆 Click a bar to focus that day. "
                "Or click the **brush** icon (top-right) and drag across the bars "
                "to filter the whole dashboard to that date range."
            )
    with right:
        _render_token_volume_section(tokens)
        _render_subagents_section(tokens, primary)

    theme.section("When the money is spent")
    tz_offset = int(
        st.number_input(
            "Punchcard UTC offset (hours)",
            min_value=-12,
            max_value=14,
            value=0,
            step=1,
            key="punchcard_tz",
            help="Shift punchcard hours from UTC (e.g. +2 for Paris / UTC+2).",
        )
    )
    heatmap = _spend_heatmap_option(tokens, primary, tz_offset=tz_offset)
    if heatmap is not None:
        echarts.render(heatmap, key="overview_heatmap", height="340px")

    # Per-session / per-prompt detail lives on the Explorer page now (no tables
    # on the analytical pages): apply a filter here, then use the "Explore →"
    # button in the filter badge to inspect the matching day/session/prompt rows.


# Render only under a real Streamlit server. A bare import (or AppTest) would
# reach the ECharts render path, and streamlit-echarts' frontend registration
# raises outside a running server (empty component registry). ``streamlit run``
# sets runtime.exists() True; a bare import leaves it False. This page is also
# excluded from the headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
