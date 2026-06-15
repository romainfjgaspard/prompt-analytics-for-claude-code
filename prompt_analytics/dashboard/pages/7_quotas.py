"""Quotas dashboard page: plan break-even, utilization gauges, reset windows.

The plan break-even (3.1, rendered here per 7.4) needs no quota snapshot at
all — it stands on the API-equivalent of the extracted history; the gauges and
trends below need `prompt-analytics snapshot` runs.

Migrated to Apache ECharts (``docs/MIGRATION-ECHARTS.md``). None of these views
is a cross-filter dimension (break-even ignores the sidebar filters entirely;
quota windows are their own axis), so nothing here emits a click. Zero Plotly
remains. Formatters are ECharts *template* strings (``{value}``/``{c}``), never
JS function strings (those only run when passed through the ``events`` dict).
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics import analytics
from prompt_analytics.analytics import Dataset
from prompt_analytics.dashboard import data, echarts, filters, theme

# Gauge zone colors: comfortable (green) / warning (amber) / critical (red),
# as fractions of the 0-100 axis (per the migration brief: vert/orange/rouge).
_GAUGE_ZONES = [[0.60, "#22C55E"], [0.85, "#F59E0B"], [1.0, "#EF4444"]]

# Quota field keys look like "five_hour" / "seven_day_sonnet" -- spell them out.
_NUM_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "thirty": "30",
}
_WINDOW_UNITS = {"minute", "hour", "day", "week", "month"}


def _pretty_field(field: str) -> str:
    """Human label for a quota field key ('seven_day_sonnet' -> '7-day · Sonnet')."""
    parts = field.split("_")
    if len(parts) >= 2 and parts[0] in _NUM_WORDS and parts[1] in _WINDOW_UNITS:
        window = f"{_NUM_WORDS[parts[0]]}-{parts[1]}"
        rest = parts[2:]
        if rest:
            return f"{window} · {' '.join(w.capitalize() for w in rest)}"
        return window
    return field.replace("_", " ").capitalize()


def _field_colors(fields: list[str]) -> dict[str, str]:
    """Stable color per quota field (shared by gauges, lines and reset marks)."""
    return {field: theme.PALETTE[i % len(theme.PALETTE)] for i, field in enumerate(sorted(fields))}


def _latest_per_field(q: pd.DataFrame) -> pd.DataFrame:
    """Return the row with the maximum ``snapshot_at`` for each ``field``."""
    work: pd.DataFrame = q.dropna(subset=["field", "snapshot_at"])
    if work.empty:
        return work
    idx = work.groupby("field")["snapshot_at"].idxmax()
    result: pd.DataFrame = work.loc[idx].reset_index(drop=True)
    return result


def _format_timedelta(delta: pd.Timedelta) -> str:
    """Format a positive timedelta as a compact human string (e.g. "3h 12m")."""
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return "—"
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def _gauge_option(label: str, value: float, color: str) -> dict[str, Any]:
    """A 0-100 gauge with green/amber/red zones for one quota field (ticks 0-100)."""
    c = echarts.colors()
    option = echarts.base_option()
    option.pop("grid", None)
    option["legend"] = {"show": False}
    option["tooltip"] = {"show": False}
    option["series"] = [
        {
            "type": "gauge",
            "min": 0,
            "max": 100,
            "splitNumber": 4,  # ticks at 0 / 25 / 50 / 75 / 100
            "radius": "92%",
            "center": ["50%", "58%"],
            "startAngle": 220,
            "endAngle": -40,
            "axisLine": {"lineStyle": {"width": 12, "color": _GAUGE_ZONES}},
            "pointer": {"width": 5, "itemStyle": {"color": color}},
            "axisTick": {"distance": -12, "length": 4, "lineStyle": {"color": c["axis"]}},
            "splitLine": {"distance": -12, "length": 12, "lineStyle": {"color": c["text"]}},
            "axisLabel": {"distance": 14, "color": c["muted"], "fontSize": 10},
            "anchor": {"show": False},
            "title": {"offsetCenter": [0, "28%"], "color": c["text"], "fontSize": 13},
            "detail": {
                "formatter": "{value}%",
                "color": c["text"],
                "fontSize": 22,
                "offsetCenter": [0, "56%"],
            },
            "data": [{"value": round(value, 1), "name": label}],
        }
    ]
    return option


def _trend_option(trend: pd.DataFrame, color_map: dict[str, str]) -> dict[str, Any]:
    """Utilization over time, with each quota window's reset overlaid (dotted)."""
    c = echarts.colors()
    option = echarts.base_option()
    option["title"] = {
        "text": "Utilization over time — dotted verticals mark window resets",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["tooltip"].update({"trigger": "axis"})
    option["xAxis"] = {
        "type": "time",
        "axisLabel": {"color": c["text"]},
        "axisLine": {"lineStyle": {"color": c["axis"]}},
    }
    yaxis = echarts.value_axis(name="Utilization (%)")
    yaxis["min"] = 0
    yaxis["max"] = 105
    # Name down the middle of the axis (not at the top) so it clears the title.
    yaxis["nameLocation"] = "middle"
    yaxis["nameGap"] = 40
    option["yAxis"] = yaxis

    # One dotted vertical per distinct reset moment, in the field's color. Only
    # resets inside the observed span are drawn, and a field with too many resets
    # in range (e.g. five_hour over months) skips its lines entirely -- they
    # would shade the whole chart.
    lo, hi = trend["snapshot_at"].min(), trend["snapshot_at"].max()
    margin = (hi - lo) * 0.05 if hi > lo else pd.Timedelta(hours=1)
    max_lines_per_field = 24

    series: list[dict[str, Any]] = []
    for field, group in trend.groupby("field"):
        field = str(field)
        color = color_map.get(field, "#9CA3AF")
        points = [
            [pd.Timestamp(ts).isoformat(), round(float(util), 1)]
            for ts, util in zip(group["snapshot_at"], group["utilization_pct"], strict=True)
        ]
        item: dict[str, Any] = {
            "name": _pretty_field(field),
            "type": "line",
            "data": points,
            "showSymbol": True,
            "symbolSize": 6,
            "lineStyle": {"color": color, "width": 2},
            "itemStyle": {"color": color},
        }
        if "resets_at" in trend.columns:
            in_range = [
                pd.Timestamp(reset)
                for reset in group["resets_at"].dropna().unique()
                if lo - margin <= pd.Timestamp(reset) <= hi + margin
            ]
            if 0 < len(in_range) <= max_lines_per_field:
                item["markLine"] = {
                    "silent": True,
                    "symbol": "none",
                    "label": {"show": False},
                    "lineStyle": {"type": "dotted", "color": theme._rgba(color, 0.5)},
                    "data": [{"xAxis": ts.isoformat()} for ts in in_range],
                }
        series.append(item)
    option["series"] = series
    return option


def _break_even_option(
    rows: list[dict[str, Any]], actual: float, factor: float, primary: str
) -> dict[str, Any]:
    """Plan prices (pro-rated to the window) as bars, your actual usage as a dashed line.

    A plan whose bar stays left of the dashed line pays for itself (green bar).
    """
    c = echarts.colors()
    plans = [str(r["plan"]) for r in rows]
    prices = [round(float(r["monthly_price_usd"]) * factor, 2) for r in rows]
    bar_colors = ["#10B981" if float(r["saving_month_usd"]) > 0 else "#9CA3AF" for r in rows]
    max_x = max([*prices, actual]) * 1.25

    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": f"Plan price vs your actual usage ({primary})",
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 8, "right": 120, "top": 48, "bottom": 40, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    xaxis = echarts.value_axis(money=True, name="USD over the window")
    xaxis["max"] = round(max_x, 2)
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.category_axis(plans)
    option["series"] = [
        {
            "type": "bar",
            "data": [
                {
                    "value": p,
                    "itemStyle": {"color": col},
                    # Literal label (no template/JS) renders verbatim.
                    "label": {
                        "show": True,
                        "position": "right",
                        "color": c["text"],
                        "formatter": f"${p:,.2f}",
                    },
                }
                for p, col in zip(prices, bar_colors, strict=True)
            ],
            "markLine": {
                "silent": True,
                "symbol": "none",
                "lineStyle": {"type": "dashed", "color": theme.PALETTE[0], "width": 2},
                "label": {
                    "color": c["text"],
                    "position": "insideEndTop",
                    "formatter": f"your usage: ${actual:,.2f}",
                },
                "data": [{"xAxis": round(actual, 2)}],
            },
        }
    ]
    return option


def _channel_bars_option(
    labels: list[str], totals: list[float], title: str, ref: float
) -> dict[str, Any]:
    """Horizontal bars of an actual windowed cost per option (cheapest = green)."""
    c = echarts.colors()
    best = min(totals)
    bar_colors = ["#10B981" if t == best else "#9CA3AF" for t in totals]
    max_x = max([*totals, ref]) * 1.25

    option = echarts.base_option()
    option["legend"] = {"show": False}
    option["title"] = {
        "text": title,
        "left": 0,
        "textStyle": {"color": c["text"], "fontSize": 16, "fontWeight": 600},
    }
    option["grid"] = {"left": 8, "right": 120, "top": 48, "bottom": 40, "containLabel": True}
    option["tooltip"].update({"trigger": "axis", "axisPointer": {"type": "shadow"}})
    xaxis = echarts.value_axis(money=True, name="USD over the window")
    xaxis["max"] = round(max_x, 2)
    option["xAxis"] = xaxis
    option["yAxis"] = echarts.category_axis(labels, inverse=True)  # cheapest on top
    option["series"] = [
        {
            "type": "bar",
            "data": [
                {
                    "value": round(t, 2),
                    "itemStyle": {"color": col},
                    "label": {
                        "show": True,
                        "position": "right",
                        "color": c["text"],
                        "formatter": f"${t:,.2f}",
                    },
                }
                for t, col in zip(totals, bar_colors, strict=True)
            ],
        }
    ]
    return option


def _render_copilot_channel(ds: Dataset) -> None:
    """What this usage actually cost, and what each Copilot subscription would."""
    rows = analytics.copilot_channel_costs(ds)
    if not rows:
        return
    span = int(rows[0]["span_days"])
    factor = span / 30  # pro-rate the monthly subscriptions to the observed window
    actual = float(rows[0]["usage_actual_usd"])
    # Window-actual cost per tier (subscription pro-rated + real overage on usage).
    labels = [str(r["label"]) for r in rows]
    totals = [float(r["total_usd"]) * factor for r in rows]

    theme.section(
        "Cost via GitHub Copilot",
        "GitHub Copilot is usage-based (GitHub AI Credits, 1 credit = $0.01). "
        "Each tier bundles a monthly usage allowance; anything beyond it is "
        "per-token overage on Copilot's grid. So a tier's *effective* cost for "
        "you is its subscription plus any overage over the included allowance.",
    )
    st.metric(f"Your actual usage on Copilot's grid ({span} days)", f"${actual:,.2f}")
    echarts.render(
        _channel_bars_option(labels, totals, "What each Copilot tier would have cost you", actual),
        key="quotas_copilot_channel",
        height="280px",
    )
    best = rows[0]
    best_sub = float(best["monthly_usd"]) * factor
    best_overage = float(best["overage_usd"]) * factor
    if best_overage <= 0.005:
        verdict = (
            f"Cheapest for you: {best['label']} — your ${actual:,.2f} of usage over {span} days "
            f"stays inside its allowance, so you'd pay just the subscription (${best_sub:,.2f} for "
            "this window)."
        )
    else:
        verdict = (
            f"Cheapest for you: {best['label']} at ${float(best['total_usd']) * factor:,.2f} over "
            f"{span} days (${best_sub:,.2f} subscription + ${best_overage:,.2f} usage overage)."
        )
    st.success(theme.md_escape(verdict))
    st.caption(
        "Actual cost over your data window — subscriptions are billed monthly but shown "
        "pro-rated to the window for a like-for-like comparison. Tiers/allowances editable in "
        "`pricing.yml` (`copilot_plans`); computed on the full extracted history (sidebar "
        "filters do not apply)."
    )


def _render_break_even(quota: pd.DataFrame, primary: str, ds: Dataset) -> None:
    """3.1 on screen (7.4): is a flat-rate plan worth it for this usage?"""
    theme.section(
        "Plan break-even",
        "Your whole history priced on the per-token grid — the actual "
        "API-equivalent of your usage — compared with each subscription plan in "
        "`pricing.yml` (plans pro-rated to your data window). Green plans pay for "
        "themselves.",
    )
    quota_rows = cast("list[dict[str, Any]]", quota.to_dict("records")) if not quota.empty else None
    result = analytics.break_even(ds, provider=primary, quota_rows=quota_rows)
    if not result.rows:
        st.info(
            "No subscription plans found — add a `plans:` section (label + "
            "monthly_usd) to pricing.yml to compare."
        )
        return

    span = analytics.observed_span_days(ds)
    factor = span / 30  # pro-rate the monthly plan prices to the observed window
    actual = float(result.rows[0]["api_equiv_month_usd"]) * factor

    st.metric(f"Your actual API-equivalent ({span} days)", f"${actual:,.2f}")
    echarts.render(
        _break_even_option(result.rows, actual, factor, primary), key="quotas_break_even"
    )

    # Table in window terms: scale the monthly money columns by the window factor.
    df = data.table_df(result)
    plan_col, api_col = f"Plan $ ({span}d)", f"Your API $ ({span}d)"
    save_col = f"Saving $ ({span}d)"
    df[plan_col] = df["Plan $/mo"].astype(float) * factor
    df[api_col] = df["Your API $/mo"].astype(float) * factor
    df[save_col] = df["Saving $/mo"].astype(float) * factor
    df = df[["Plan", plan_col, api_col, "vs plan", save_col]]
    st.dataframe(
        df.style.format(
            {plan_col: "${:,.2f}", api_col: "${:,.2f}", "vs plan": "x{:.2f}", save_col: "${:,.2f}"},
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )

    # Verdict in window terms (the ratio / sign is the same as monthly).
    worth = [r for r in result.rows if float(r["saving_month_usd"]) > 0]
    if worth:
        best = max(worth, key=lambda r: float(r["saving_month_usd"]))
        price = float(best["monthly_price_usd"]) * factor
        st.success(
            theme.md_escape(
                f"At this rate the {best['plan']} plan (${price:,.2f} over these {span} days, "
                f"${float(best['monthly_price_usd']):,.0f}/mo) pays for itself — your "
                f"${actual:,.2f} of API-equivalent is x{best['vs_plan']} the price."
            )
        )
    else:
        cheapest = result.rows[0]  # sorted by price ascending
        price = float(cheapest["monthly_price_usd"]) * factor
        st.warning(
            theme.md_escape(
                f"At this rate no plan pays off: even {cheapest['plan']} (${price:,.2f} over "
                f"these {span} days) costs more than your ${actual:,.2f} of API-equivalent — "
                "pay-as-you-go API is cheaper for this volume."
            )
        )
    # Quota-window enrichment note (utilization %, not a money projection).
    for note in result.notes[1:]:
        if "utilization" in note or "snapshot" in note:
            st.caption(theme.md_escape(note))
    st.caption(
        f"Actual cost over your {span}-day data window (sidebar filters do not apply); "
        "subscriptions are billed monthly (list prices, editable in `pricing.yml`) and shown "
        "pro-rated to the window for a like-for-like comparison."
    )


def _render_quota_windows(q: pd.DataFrame) -> None:
    """Current utilization gauges, utilization-over-time, and next-reset countdowns.

    Uses ``return`` (not ``st.stop``) when there is nothing to show so the cost
    sections below it still render — the break-even needs no quota snapshot.
    """
    theme.section("Quota windows")
    cfg = data.load_config()
    if not cfg.get("features", {}).get("quota_snapshot", False):
        st.info("Quota snapshots disabled in config.")
        return

    if q.empty:
        st.info("No quota snapshots yet. Run `prompt-analytics snapshot`.")
        return

    latest = _latest_per_field(q)
    colors = _field_colors([str(f) for f in latest.get("field", pd.Series(dtype=str))])

    # --- Current utilization gauges (one column per field). ---
    st.subheader("Current utilization")
    if latest.empty:
        st.info("No valid snapshots to display.")
    else:
        cols = st.columns(len(latest))
        for col, (_, row) in zip(cols, latest.iterrows(), strict=True):
            value = row.get("utilization_pct")
            field = str(row["field"])
            with col:
                if pd.isna(value):
                    st.metric(_pretty_field(field), "—")
                else:
                    echarts.render(
                        _gauge_option(
                            _pretty_field(field),
                            float(value),
                            colors.get(field, theme.PALETTE[0]),
                        ),
                        key=f"quotas_gauge_{field}",
                        height="240px",
                    )

    # --- Utilization over time per field, reset windows overlaid. ---
    trend = q.dropna(subset=["snapshot_at", "utilization_pct", "field"]).sort_values("snapshot_at")
    if trend.empty:
        st.info("No time-series data available.")
    else:
        echarts.render(_trend_option(trend, colors), key="quotas_trend", height="420px")
        st.caption(
            "Each dotted vertical is a quota-window reset (`resets_at`): "
            "utilization climbs within a window and drops after the reset."
        )

    # --- Next reset countdown per field. ---
    st.subheader("Next reset")
    if latest.empty:
        st.info("No reset information available.")
    else:
        now_utc = pd.Timestamp.now(tz="UTC")
        cols = st.columns(len(latest))
        for col, (_, row) in zip(cols, latest.iterrows(), strict=True):
            resets_at = row.get("resets_at")
            label = _pretty_field(str(row["field"]))
            with col:
                if pd.isna(resets_at):
                    st.metric(label, "—")
                else:
                    delta = pd.Timestamp(resets_at) - now_utc
                    st.metric(label, _format_timedelta(delta))


def main() -> None:
    """Render the Quotas page (windows first, then the cost story)."""
    st.title("Quotas & plan break-even")
    # This page has no sidebar filters; keep any selection from other pages alive.
    filters.persist_filters()

    q = data.load_all()["quota_log"]
    primary = data.primary_provider()
    ds = data.load_dataset()
    _render_quota_windows(q)
    _render_break_even(q, primary, ds)
    _render_copilot_channel(ds)


# Render only under a real Streamlit server: streamlit-echarts cannot register
# its frontend under a bare import / AppTest (empty component registry); only
# under `streamlit run` (runtime.exists() True). This page is excluded from the
# headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
