"""Tests for the dashboard data layer, filters, tz bounds and a headless smoke.

These exercise phase 8 (the dashboard now consumes ``analytics.py``) and the
9.x demo dataset, and they pin the 8.1 tz-naive/tz-aware crash as a
non-regression. The Streamlit pages are run headless through
``streamlit.testing.v1.AppTest`` (no browser, no server).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("streamlit")

import pandas as pd  # noqa: E402

from prompt_analytics import analytics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo_data"
DASHBOARD_DIR = REPO_ROOT / "prompt_analytics" / "dashboard"
PAGES_DIR = DASHBOARD_DIR / "pages"


pytestmark = pytest.mark.skipif(
    not DEMO_DIR.exists(), reason="demo_data/ not generated (run scripts/generate_demo_data.py)"
)


@pytest.fixture
def demo_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the dashboard at the committed demo dataset and clear caches."""
    monkeypatch.setenv("CCA_DEMO", "1")
    monkeypatch.delenv("CCA_DATA_DIR", raising=False)
    monkeypatch.delenv("PROMPT_ANALYTICS_OUTPUT_DIR", raising=False)
    from prompt_analytics.dashboard import data as data_mod

    data_mod._load_cached.clear()
    return DEMO_DIR


# ---------------------------------------------------------------------------
# Demo-mode detection (7.1: env var locally, st.secrets on Streamlit Cloud).
# ---------------------------------------------------------------------------


def test_is_demo_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from prompt_analytics.dashboard import data as data_mod

    monkeypatch.setenv("CCA_DEMO", "1")
    assert data_mod.is_demo() is True


def test_is_demo_from_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit Community Cloud delivers CCA_DEMO as a secret, not an env var."""
    from prompt_analytics.dashboard import data as data_mod

    monkeypatch.delenv("CCA_DEMO", raising=False)
    monkeypatch.setattr("streamlit.secrets", {"CCA_DEMO": "1"})
    assert data_mod.is_demo() is True


def test_is_demo_false_without_env_or_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var and reading st.secrets raises (no secrets file) -> not demo."""
    from prompt_analytics.dashboard import data as data_mod

    class _Raising:
        def get(self, *_args: object) -> object:
            raise RuntimeError("no secrets file configured")

    monkeypatch.delenv("CCA_DEMO", raising=False)
    monkeypatch.setattr("streamlit.secrets", _Raising())
    assert data_mod.is_demo() is False


# ---------------------------------------------------------------------------
# Aggregations on the demo dataset.
# ---------------------------------------------------------------------------


def test_demo_dataset_loads_expected_shape() -> None:
    ds = analytics.dataset_from_csvs(DEMO_DIR)
    assert len(ds.sessions) == 80
    assert 750 <= len(ds.prompts) <= 850
    assert ds.tokens  # non-empty
    assert ds.categories  # categorize filled


def test_demo_summary_projects_and_cost() -> None:
    ds = analytics.dataset_from_csvs(DEMO_DIR)
    result = analytics.summary(ds)
    metrics = {row["metric"]: row["value"] for row in result.rows}
    assert metrics["Sessions"] == 80
    assert metrics["Projects"] == 5
    # Cost lines are formatted strings like "$178.14"; parse and check positive.
    anthropic = float(metrics["Cost (anthropic)"].lstrip("$").replace(",", ""))
    assert anthropic > 0


def test_demo_session_depth_opening_is_most_expensive() -> None:
    """The whole point of the depth analysis: depth 1 costs the most."""
    ds = analytics.dataset_from_csvs(DEMO_DIR)
    rows = analytics.session_depth(ds, "anthropic").rows
    assert rows
    opening = rows[0]
    assert opening["depth"] == "1"
    assert opening["vs_depth_1"] == 1.0
    assert all(r["vs_depth_1"] <= 1.0 for r in rows[1:])


def test_demo_by_project_shares_sum_to_100() -> None:
    ds = analytics.dataset_from_csvs(DEMO_DIR)
    rows = analytics.by_project(ds, "anthropic").rows
    assert len(rows) == 5
    assert rows[-1]["cumulative_pct"] == pytest.approx(100.0, abs=0.2)


def test_dashboard_costs_match_analytics(demo_env: Path) -> None:
    """The dashboard's per-token cost column equals the analytics total (8.2)."""
    from prompt_analytics.dashboard import data as data_mod

    frames = data_mod.load_all()
    tokens = frames["tokens"]
    frame_total = float(tokens[data_mod.cost_col("anthropic")].sum())

    ds = analytics.dataset_from_csvs(DEMO_DIR)
    engine = analytics.CostEngine("anthropic")
    cli_total = sum(analytics._prompt_costs(ds, engine).values())
    assert frame_total == pytest.approx(cli_total, rel=1e-9)


def test_demo_dataset_contains_session_overhead() -> None:
    """The demo set must hold pseudo-prompt rows (tokens.csv only) so the
    post-filter parity test below actually exercises the N1 cascade."""
    ds = analytics.dataset_from_csvs(DEMO_DIR)
    real_ids = {row["prompt_id"] for row in ds.prompts}
    pseudo_rows = [row for row in ds.tokens if row["prompt_id"] not in real_ids]
    assert pseudo_rows, "demo_data has no pseudo-prompt token rows"
    assert all(":_" in row["prompt_id"] for row in pseudo_rows)


def test_dashboard_costs_match_analytics_post_filters(
    demo_env: Path, no_session_state: None
) -> None:
    """Parity must hold AFTER apply_filters with no active filter (N1):
    pseudo-prompt token rows (session overhead) used to be dropped there."""
    from prompt_analytics.dashboard import data as data_mod
    from prompt_analytics.dashboard import filters

    frames = filters.apply_filters(data_mod.load_all())
    frame_total = float(frames["tokens"][data_mod.cost_col("anthropic")].sum())

    ds = analytics.dataset_from_csvs(DEMO_DIR)
    engine = analytics.CostEngine("anthropic")
    cli_total = sum(analytics._prompt_costs(ds, engine).values())
    assert frame_total == pytest.approx(cli_total, rel=1e-9)


# ---------------------------------------------------------------------------
# filters.apply_filters.
# ---------------------------------------------------------------------------


def _frames() -> dict[str, pd.DataFrame]:
    prompts = pd.DataFrame(
        {
            "prompt_id": ["p1", "p2", "p3"],
            "session_id": ["s1", "s1", "s2"],
            "model": ["claude-opus-4-8", "claude-haiku-4-5", "claude-opus-4-8"],
            "project": ["alpha", "alpha", "beta"],
            "timestamp": pd.to_datetime(
                ["2026-05-01T10:00:00Z", "2026-05-03T10:00:00Z", "2026-05-10T10:00:00Z"],
                utc=True,
            ),
        }
    )
    # s2:_continuation is a pseudo-prompt: in tokens only, never in prompts (N1).
    tokens = pd.DataFrame(
        {
            "prompt_id": ["p1", "p2", "p3", "s2:_continuation"],
            "session_id": ["s1", "s1", "s2", "s2"],
            "token_type": ["input", "input", "input", "cache_read"],
            "token_count": [1, 2, 3, 4],
        }
    )
    sessions = pd.DataFrame({"session_id": ["s1", "s2"], "project": ["alpha", "beta"]})
    return {"prompts": prompts, "tokens": tokens, "sessions": sessions}


@pytest.fixture
def no_session_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make apply_filters ignore st.session_state (no runtime in unit tests)."""
    from prompt_analytics.dashboard import filters

    monkeypatch.setattr(
        filters,
        "get_filter_state",
        lambda: {"date_range": None, "models": None, "projects": None, "categories": None},
    )


def test_apply_filters_by_model(no_session_state: None) -> None:
    from prompt_analytics.dashboard import filters

    out = filters.apply_filters(_frames(), models=["claude-opus-4-8"])
    assert set(out["prompts"]["prompt_id"]) == {"p1", "p3"}
    # Cascade to tokens and sessions; the pseudo-prompt follows its session.
    assert set(out["tokens"]["prompt_id"]) == {"p1", "p3", "s2:_continuation"}
    assert set(out["sessions"]["session_id"]) == {"s1", "s2"}


def test_apply_filters_all_models_selected_keeps_model_less_prompt(
    no_session_state: None,
) -> None:
    """Selecting every model is "all", not a filter: a prompt with a blank model
    (no billed assistant turn) must still be counted, so the KPI total matches the
    extract total. A proper subset, however, drops it."""
    from prompt_analytics.dashboard import filters

    frames = _frames()
    prompts = pd.concat(
        [
            frames["prompts"],
            pd.DataFrame(
                {
                    "prompt_id": ["p4"],
                    "session_id": ["s1"],
                    "model": [""],  # model-less: stripped from the filter chips
                    "project": ["alpha"],
                    "timestamp": pd.to_datetime(["2026-05-04T10:00:00Z"], utc=True),
                }
            ),
        ],
        ignore_index=True,
    )
    frames["prompts"] = prompts
    all_models = ["claude-opus-4-8", "claude-haiku-4-5"]

    kept = filters.apply_filters(frames, models=all_models)
    assert set(kept["prompts"]["prompt_id"]) == {"p1", "p2", "p3", "p4"}

    narrowed = filters.apply_filters(frames, models=["claude-opus-4-8"])
    assert set(narrowed["prompts"]["prompt_id"]) == {"p1", "p3"}


def test_apply_filters_by_project(no_session_state: None) -> None:
    from prompt_analytics.dashboard import filters

    out = filters.apply_filters(_frames(), projects=["beta"])
    assert set(out["prompts"]["prompt_id"]) == {"p3"}
    assert set(out["tokens"]["prompt_id"]) == {"p3", "s2:_continuation"}
    assert set(out["sessions"]["session_id"]) == {"s2"}


def test_apply_filters_keeps_session_overhead_without_filters(no_session_state: None) -> None:
    """No active filter must be an identity on tokens: pseudo-prompt rows
    (session overhead, in tokens.csv only) used to be silently dropped (N1)."""
    from prompt_analytics.dashboard import filters

    out = filters.apply_filters(_frames())
    assert set(out["tokens"]["prompt_id"]) == {"p1", "p2", "p3", "s2:_continuation"}
    assert int(out["tokens"]["token_count"].sum()) == 10


def test_apply_filters_drops_overhead_of_filtered_out_sessions(no_session_state: None) -> None:
    """Session overhead follows its session: filtering s2 out drops its tail."""
    from prompt_analytics.dashboard import filters

    out = filters.apply_filters(_frames(), projects=["alpha"])
    assert set(out["tokens"]["prompt_id"]) == {"p1", "p2"}


def test_apply_filters_by_date_range(no_session_state: None) -> None:
    from prompt_analytics.dashboard import filters

    out = filters.apply_filters(
        _frames(),
        date_range=(datetime.date(2026, 5, 2), datetime.date(2026, 5, 5)),
    )
    assert set(out["prompts"]["prompt_id"]) == {"p2"}


def test_apply_filters_by_category(no_session_state: None) -> None:
    """The category filter (5.4) restricts prompts then cascades like the rest."""
    from prompt_analytics.dashboard import filters

    frames = _frames()
    frames["prompts"]["category"] = ["debug", "debug", "refactor"]
    out = filters.apply_filters(frames, categories=["refactor"])
    assert set(out["prompts"]["prompt_id"]) == {"p3"}
    # Cascade reaches tokens (incl. the pseudo-prompt via its session) + sessions.
    assert set(out["tokens"]["prompt_id"]) == {"p3", "s2:_continuation"}
    assert set(out["sessions"]["session_id"]) == {"s2"}


# ---------------------------------------------------------------------------
# theme.project_color_map: distinct, name -> color, shared from a universe (5.2).
# ---------------------------------------------------------------------------


def test_project_color_map_is_distinct() -> None:
    """Real projects get different hues (no name-hash collisions for small N)."""
    from prompt_analytics.dashboard import theme

    mapping = theme.project_color_map(["alpha", "beta", "gamma", "delta", "epsilon"])
    hues = [mapping[p] for p in ("alpha", "beta", "gamma", "delta", "epsilon")]
    assert len(set(hues)) == len(hues)


def test_project_color_map_stable_for_a_fixed_universe() -> None:
    """Built from the same universe, the map is identical run to run -- which is
    how the Sessions page keeps a hue stable across its three charts/filters."""
    from prompt_analytics.dashboard import theme

    universe = ["data-pipeline", "webapp-frontend", "ml-experiments"]
    assert theme.project_color_map(universe) == theme.project_color_map(reversed(universe))


def test_project_color_map_specials_always_present_and_grey() -> None:
    from prompt_analytics.dashboard import theme

    mapping = theme.project_color_map(["alpha"])  # specials not even passed
    assert mapping["(session overhead)"] == "#9CA3AF"
    assert mapping["(unknown)"] == "#D1D5DB"
    assert mapping["alpha"] not in {"#9CA3AF", "#D1D5DB"}


# ---------------------------------------------------------------------------
# theme.language_color_map: curated hues for known languages, stable cycle else.
# ---------------------------------------------------------------------------


def test_language_color_map_uses_curated_hues() -> None:
    from prompt_analytics.dashboard import theme

    mapping = theme.language_color_map(["Python", "TypeScript"])
    assert mapping["Python"] == theme.LANGUAGE_COLORS["Python"]
    assert mapping["TypeScript"] == theme.LANGUAGE_COLORS["TypeScript"]


def test_language_color_map_tooling_bucket_is_grey_and_distinct() -> None:
    from prompt_analytics.dashboard import theme

    mapping = theme.language_color_map(["Python", "(other tooling)"])
    assert mapping["(other tooling)"] == "#64748B"
    assert mapping["Python"] != mapping["(other tooling)"]


def test_language_color_map_unknown_languages_get_distinct_stable_hues() -> None:
    from prompt_analytics.dashboard import theme

    langs = ["zig", "nim", "crystal"]  # none curated
    mapping = theme.language_color_map(langs)
    hues = [mapping[lang] for lang in langs]
    assert len(set(hues)) == len(hues)  # distinct for small N
    assert theme.language_color_map(langs) == theme.language_color_map(reversed(langs))  # stable


def test_box_stats_uses_p5_p95_whiskers() -> None:
    """Robust whiskers: the 5-number summary uses p5/p95, not min/max."""
    from prompt_analytics.dashboard import data as data_mod

    assert data_mod.box_stats(range(101)) == [5.0, 25.0, 50.0, 75.0, 95.0]


def test_box_cap_clips_above_tallest_box_and_counts_overflow() -> None:
    """The cap is 25% above the tallest Q3 (box top); n_above counts what spills over."""
    from prompt_analytics.dashboard import data as data_mod

    # Q3 = 4.0 -> y_max = 4.0 * 1.25 = 5.0; only the 1000 lies above it.
    y_max, n_above = data_mod.box_cap([[1, 2, 3, 4, 5, 1000]], [[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert y_max == 5.0
    assert n_above == 1


def test_box_cap_returns_none_when_nothing_to_clip() -> None:
    """All-zero data has no positive whisker, so there is nothing to clip."""
    from prompt_analytics.dashboard import data as data_mod

    assert data_mod.box_cap([[0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0]]) == (None, 0)


def test_auto_granularity_by_span() -> None:
    """Default grain widens with the observed span: Day -> Week -> Month."""
    from prompt_analytics.dashboard import data as data_mod

    assert (
        data_mod.auto_granularity(pd.to_datetime(["2026-06-01", "2026-06-20"], utc=True)) == "Day"
    )
    assert (
        data_mod.auto_granularity(pd.to_datetime(["2026-01-01", "2026-04-01"], utc=True)) == "Week"
    )
    assert (
        data_mod.auto_granularity(pd.to_datetime(["2026-01-01", "2026-09-01"], utc=True)) == "Month"
    )


def test_to_period_floors_to_week_monday_and_month_start() -> None:
    """Week buckets land on Mondays; month buckets on the 1st; Day is unchanged."""
    from prompt_analytics.dashboard import data as data_mod

    s = pd.Series(pd.to_datetime(["2026-06-10", "2026-06-11", "2026-07-02"], utc=True))
    week = data_mod.to_period(s, "Week")
    assert set(week.dt.weekday.unique()) == {0}  # every bucket is a Monday
    assert (week <= s.dt.normalize()).all()  # the week start is on/before the day
    month = data_mod.to_period(s, "Month")
    assert list(month.dt.strftime("%Y-%m")) == ["2026-06", "2026-06", "2026-07"]
    assert set(month.dt.day.unique()) == {1}
    assert list(data_mod.to_period(s, "Day").dt.strftime("%Y-%m-%d")) == [
        "2026-06-10",
        "2026-06-11",
        "2026-07-02",
    ]


# ---------------------------------------------------------------------------
# Hero numbers (7.1) and TableResult -> DataFrame (7.4).
# ---------------------------------------------------------------------------


def test_context_rent_share_hand_computed() -> None:
    """Rent = cache reads + cache writes over the total API-equivalent cost."""
    from prompt_analytics.dashboard import data as data_mod

    tokens = pd.DataFrame(
        {
            "token_type": ["cache_read", "cache_write_1h", "cache_write_5m", "output", "input"],
            "cost_anthropic_usd": [50.0, 25.0, 5.0, 15.0, 5.0],
        }
    )
    assert data_mod._context_rent_share(tokens, "anthropic") == pytest.approx(80.0)
    # No cost at all -> no hero (never a division by zero).
    empty = pd.DataFrame({"token_type": ["input"], "cost_anthropic_usd": [0.0]})
    assert data_mod._context_rent_share(empty, "anthropic") is None


def test_table_df_maps_keys_to_labels_in_order() -> None:
    from prompt_analytics.dashboard import data as data_mod

    result = analytics.TableResult(
        "t",
        [analytics.Column("a", "A label", "int"), analytics.Column("b", "B", "money")],
        [{"a": 1, "b": 2.5}],
    )
    df = data_mod.table_df(result)
    assert list(df.columns) == ["A label", "B"]
    assert df.iloc[0]["A label"] == 1
    # Empty rows still yield the labeled columns (stable downstream rendering).
    empty = data_mod.table_df(analytics.TableResult("t", result.columns, []))
    assert list(empty.columns) == ["A label", "B"]


# ---------------------------------------------------------------------------
# tz bounds: non-regression of the tz-naive/tz-aware crash (8.1).
# ---------------------------------------------------------------------------


def test_available_date_bounds_mixed_tz_does_not_crash() -> None:
    """sessions.start_date naive + prompts.timestamp aware used to raise."""
    from prompt_analytics.dashboard import app

    frames = {
        "prompts": pd.DataFrame({"timestamp": pd.to_datetime(["2026-05-01T10:00:00Z"], utc=True)}),
        # Deliberately tz-naive, as sessions.csv start_date used to be parsed.
        "sessions": pd.DataFrame({"start_date": pd.to_datetime(["2026-04-28"])}),
    }
    bounds = app._available_date_bounds(frames)
    assert bounds is not None
    lo, hi = bounds
    assert lo == datetime.date(2026, 4, 28)
    assert hi == datetime.date(2026, 5, 1)


# ---------------------------------------------------------------------------
# Headless Streamlit smoke test (every page runs without an exception).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script",
    [
        # app.py and pages 1_overview / 2_models / 3_prompts / 4_session_depth /
        # 5_sessions / 6_optimize / 7_quotas / 11_explorer are intentionally
        # absent: they now render ECharts in main() after the data guards, and
        # streamlit-echarts cannot register under AppTest (empty component
        # registry) -- only under a real `streamlit run`. Their calculations are
        # covered by the data.py / filters.py unit tests below (we assert the
        # numbers, not the rendering). (8_providers was removed: the cost
        # comparison lives on Quotas now.)
        #
        # 10_how_it_works is the ONLY analytics page left here: it carries no
        # ECharts (native widgets + the pricing.yml cost tables only), so it both
        # runs headless and follows the light/dark toggle on its own. ECharts
        # screenshots are captured from a real browser, not generated offline
        # (the Plotly/kaleido scripts/generate_screenshots.py was removed).
        "pages/10_how_it_works.py",
    ],
)
def test_streamlit_pages_run_headless(demo_env: Path, script: str) -> None:
    at = pytest.importorskip("streamlit.testing.v1")
    app_test = at.AppTest.from_file(str(DASHBOARD_DIR / script), default_timeout=30)
    app_test.run()
    assert not app_test.exception, f"{script} raised: {app_test.exception}"


# ---------------------------------------------------------------------------
# Unit tests for data.py and filters.py transformations (8.5).
# These assert calculated values, not just "no exception", so that calculation
# bugs like N1 (pseudo-prompt rows silently dropped) surface in the test suite.
# ---------------------------------------------------------------------------


def test_add_cost_columns_linearity() -> None:
    """Doubling token_count doubles the cost (no fixed-per-call overhead)."""
    from prompt_analytics.dashboard import data as data_mod

    tokens = pd.DataFrame(
        {
            "model": ["claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"],
            "token_type": ["input", "input"],
            "token_count": [1_000, 2_000],
        }
    )
    data_mod._add_cost_columns(tokens, ["anthropic"], None)
    col = data_mod._cost_column_name("anthropic")
    assert col in tokens.columns
    cost_1k = float(tokens.iloc[0][col])
    cost_2k = float(tokens.iloc[1][col])
    assert cost_1k > 0
    assert cost_2k == pytest.approx(2 * cost_1k, rel=1e-9)


def test_add_cost_columns_matches_engine_directly() -> None:
    """_add_cost_columns must produce the same cost as CostEngine.cost()."""
    from prompt_analytics import analytics
    from prompt_analytics.dashboard import data as data_mod

    model = "claude-haiku-4-5-20251001"
    count = 5_000
    tokens = pd.DataFrame({"model": [model], "token_type": ["cache_read"], "token_count": [count]})
    data_mod._add_cost_columns(tokens, ["anthropic"], None)
    col = data_mod._cost_column_name("anthropic")
    frame_cost = float(tokens.iloc[0][col])
    engine_cost = analytics.CostEngine("anthropic").cost(model, "cache_read", count)
    assert frame_cost == pytest.approx(engine_cost, rel=1e-9)


def test_add_cost_columns_empty_df_creates_column() -> None:
    """_add_cost_columns on an empty frame must still add the column (no KeyError)."""
    from prompt_analytics.dashboard import data as data_mod

    tokens = pd.DataFrame(columns=["model", "token_type", "token_count"])
    data_mod._add_cost_columns(tokens, ["anthropic"], None)
    assert data_mod._cost_column_name("anthropic") in tokens.columns


def test_build_frames_categories_merged_onto_prompts(demo_env: Path) -> None:
    """_build_frames must merge Dataset.categories onto the prompts frame."""
    from prompt_analytics import analytics
    from prompt_analytics.dashboard import data as data_mod

    ds = analytics.dataset_from_csvs(DEMO_DIR)
    frames = data_mod._build_frames(ds, ["anthropic"], DEMO_DIR)
    assert "category" in frames["prompts"].columns
    assert not frames["prompts"]["category"].isna().all()


def test_build_frames_per_prompt_cost_equals_token_sum(demo_env: Path) -> None:
    """The per-prompt cost column on prompts must equal the sum of its token rows."""
    from prompt_analytics import analytics
    from prompt_analytics.dashboard import data as data_mod

    ds = analytics.dataset_from_csvs(DEMO_DIR)
    frames = data_mod._build_frames(ds, ["anthropic"], DEMO_DIR)
    tokens = frames["tokens"]
    prompts = frames["prompts"]
    col = data_mod.cost_col("anthropic")

    pid = prompts["prompt_id"].iloc[0]
    token_sum = float(tokens.loc[tokens["prompt_id"] == pid, col].sum())
    prompt_cost = float(prompts.loc[prompts["prompt_id"] == pid, col].iloc[0])
    assert prompt_cost == pytest.approx(token_sum, rel=1e-9)


def test_build_frames_date_columns_are_utc_aware(demo_env: Path) -> None:
    """prompts.date and tokens.date must be tz-aware UTC (no naive/aware mixing)."""
    from prompt_analytics import analytics
    from prompt_analytics.dashboard import data as data_mod

    ds = analytics.dataset_from_csvs(DEMO_DIR)
    frames = data_mod._build_frames(ds, ["anthropic"], DEMO_DIR)
    for frame_name in ("prompts", "tokens"):
        frame = frames[frame_name]
        if "date" in frame.columns and not frame["date"].isna().all():
            dtype = frame["date"].dtype
            assert isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC", (
                f"{frame_name}.date is not tz-aware UTC (got {dtype})"
            )


def test_impact_to_iso_handles_date_datetime_str_none() -> None:
    """impact._to_iso normalizes the pivot value the date widget / a suggestion stores."""
    from prompt_analytics.dashboard import impact

    assert impact._to_iso(datetime.date(2026, 6, 5)) == "2026-06-05"
    assert impact._to_iso(datetime.datetime(2026, 6, 5, 14, 30)) == "2026-06-05"
    assert impact._to_iso("2026-06-05") == "2026-06-05"
    assert impact._to_iso("not a date") is None
    assert impact._to_iso(None) is None
    assert impact._to_iso("") is None


def test_available_date_bounds_correct_dates(no_session_state: None) -> None:
    """available_date_bounds returns the min and max date across both frames."""
    from prompt_analytics.dashboard import filters

    frames = {
        "prompts": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-05-10T00:00:00Z", "2026-05-20T00:00:00Z"], utc=True
                )
            }
        ),
        "sessions": pd.DataFrame(
            {"start_date": pd.to_datetime(["2026-05-01T00:00:00Z"], utc=True)}
        ),
    }
    bounds = filters.available_date_bounds(frames)
    assert bounds is not None
    lo, hi = bounds
    assert lo == datetime.date(2026, 5, 1)
    assert hi == datetime.date(2026, 5, 20)


def test_options_extracts_models_projects_categories(no_session_state: None) -> None:
    """_options must return sorted, distinct values for each filter dimension."""
    from prompt_analytics.dashboard import filters

    frames = _frames()
    frames["prompts"]["category"] = ["debug", "refactor", "debug"]
    opts = filters._options(frames)
    assert opts["models"] == ["claude-haiku-4-5", "claude-opus-4-8"]
    assert opts["projects"] == ["alpha", "beta"]
    assert opts["categories"] == ["debug", "refactor"]


def _patch_state(monkeypatch: pytest.MonkeyPatch, **state: object) -> None:
    """Override ``filters.get_filter_state`` with a fixed dict for the test."""
    from prompt_analytics.dashboard import filters

    base: dict[str, object] = {
        "date_range": None,
        "models": None,
        "projects": None,
        "categories": None,
        "xf_date_range": None,
        "xf_models": None,
        "xf_projects": None,
        "xf_categories": None,
    }
    base.update(state)
    monkeypatch.setattr(filters, "get_filter_state", lambda: base)


def test_xf_parts_reports_chart_click_drill(monkeypatch: pytest.MonkeyPatch) -> None:
    """The badge summarizes only the chart-click drill (xf_*), not the sidebar."""
    from prompt_analytics.dashboard import filters

    _patch_state(monkeypatch, xf_projects=["beta"], xf_models=["claude-opus-4-8"])
    parts = filters._xf_parts()
    assert "beta" in parts
    # Model drill renders through theme.model_label, not the raw id.
    assert any(p != "claude-opus-4-8" and "Opus" in p for p in parts)


def test_xf_parts_ignores_persistent_sidebar_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sidebar selection (even a proper subset) must NOT raise the badge."""
    from prompt_analytics.dashboard import filters

    _patch_state(monkeypatch, models=["claude-opus-4-8"], projects=["beta"])
    assert filters._xf_parts() == []


def test_apply_filters_ands_drill_on_top_of_sidebar(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chart-click drill (xf_*) narrows on top of the sidebar selection."""
    from prompt_analytics.dashboard import filters

    # Sidebar leaves everything; a drill to project beta restricts to p3.
    _patch_state(monkeypatch, xf_projects=["beta"])
    out = filters.apply_filters(_frames())
    assert set(out["prompts"]["prompt_id"]) == {"p3"}


def test_apply_filters_by_prompt_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompts-per-session drill keeps only sessions of that prompt count.

    In ``_frames`` s1 has two prompts (p1, p2) and s2 has one (p3), so a drill to
    ``1`` keeps s2 (and its session-overhead tail), and ``2`` keeps s1.
    """
    from prompt_analytics.dashboard import filters

    _patch_state(monkeypatch, xf_prompt_count=[1])
    one = filters.apply_filters(_frames())
    assert set(one["prompts"]["prompt_id"]) == {"p3"}
    assert set(one["tokens"]["prompt_id"]) == {"p3", "s2:_continuation"}

    _patch_state(monkeypatch, xf_prompt_count=[2])
    two = filters.apply_filters(_frames())
    assert set(two["prompts"]["prompt_id"]) == {"p1", "p2"}


def test_xf_parts_reports_prompt_count_drill(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompts-per-session drill shows in the badge (singular for 1)."""
    from prompt_analytics.dashboard import filters

    _patch_state(monkeypatch, xf_prompt_count=[1])
    assert "1 prompt/session" in filters._xf_parts()
    _patch_state(monkeypatch, xf_prompt_count=[3])
    assert "3 prompts/session" in filters._xf_parts()


# ---------------------------------------------------------------------------
# Refresh-data button pipeline (sidebar "Refresh data").
# ---------------------------------------------------------------------------


def test_refresh_data_disabled_on_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    """refresh_data must refuse to run against the bundled demo dataset."""
    from prompt_analytics.dashboard import data as data_mod

    monkeypatch.setenv("CCA_DEMO", "1")
    with pytest.raises(RuntimeError, match="demo"):
        data_mod.refresh_data()


def test_refresh_data_runs_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """refresh_data runs extract -> snapshot -> *semantic* categorize and summarizes."""
    from dataclasses import dataclass

    from prompt_analytics import categorize, extract, snapshot
    from prompt_analytics.dashboard import data as data_mod

    monkeypatch.delenv("CCA_DEMO", raising=False)
    monkeypatch.setattr(data_mod, "data_dir", lambda: tmp_path)

    @dataclass
    class _Report:
        prompts: int = 42
        sessions: int = 7

    calls: dict[str, object] = {}

    def fake_extract(directory: Path, **_kwargs: object) -> _Report:
        calls["extract"] = directory
        return _Report()

    def fake_snapshot(directory: Path) -> int:
        calls["snapshot"] = directory
        return 0

    def fake_categorize(*, output_dir: str, **kwargs: object) -> int:
        calls["categorize"] = output_dir
        calls["use_llm"] = kwargs.get("use_llm", False)
        calls["use_semantic"] = kwargs.get("use_semantic", False)
        return 42

    monkeypatch.setattr(extract, "run_extract", fake_extract)
    monkeypatch.setattr(snapshot, "run_snapshot", fake_snapshot)
    monkeypatch.setattr(categorize, "run_categorize", fake_categorize)

    summary = data_mod.refresh_data()

    assert calls["extract"] == tmp_path
    assert calls["snapshot"] == tmp_path
    assert calls["categorize"] == str(tmp_path)
    assert calls["use_llm"] is False  # never the LLM -- no API cost from the button
    assert calls["use_semantic"] is True  # semantic is the dashboard default
    assert "42 prompts" in summary
    assert "7 sessions" in summary


def test_refresh_data_falls_back_to_heuristic_when_embedder_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the embedding model can't load (offline / model2vec missing) the
    semantic run raises RuntimeError; refresh_data must fall back to the heuristic
    classifier rather than failing the whole refresh."""
    from dataclasses import dataclass

    from prompt_analytics import categorize, extract, snapshot
    from prompt_analytics.dashboard import data as data_mod

    monkeypatch.delenv("CCA_DEMO", raising=False)
    monkeypatch.setattr(data_mod, "data_dir", lambda: tmp_path)

    @dataclass
    class _Report:
        prompts: int = 3
        sessions: int = 1

    modes: list[bool] = []

    def fake_categorize(*, output_dir: str, **kwargs: object) -> int:
        semantic = bool(kwargs.get("use_semantic", False))
        modes.append(semantic)
        if semantic:
            raise RuntimeError("Could not load the static embedding model")
        return 3

    monkeypatch.setattr(extract, "run_extract", lambda directory, **_k: _Report())
    monkeypatch.setattr(snapshot, "run_snapshot", lambda directory: 0)
    monkeypatch.setattr(categorize, "run_categorize", fake_categorize)

    summary = data_mod.refresh_data()

    assert modes == [True, False]  # tried semantic, then fell back to heuristic
    assert "3 prompts" in summary
