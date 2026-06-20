"""Tests for the CLI (11.6): dispatch, exit codes, --format, exclusive flags.

Every test goes through :func:`prompt_analytics.cli.main` with a real argv,
on the ``fake_claude`` sandbox (fixtures under a fake ``~/.claude/projects``).
"""

from __future__ import annotations

import csv
import io
import json
import runpy
import sys

import pytest

from prompt_analytics.cli import main


def _args(fake_claude, *extra):
    return [*extra, "--output-dir", str(fake_claude.out)]


# ---------------------------------------------------------------------------
# Module entry point (python -m prompt_analytics) — exit codes propagate (A2).
# ---------------------------------------------------------------------------


def test_module_entry_point_propagates_exit_code(monkeypatch):
    """`python -m prompt_analytics --help` runs main() and exits 0 (A2)."""
    monkeypatch.setattr(sys, "argv", ["prompt-analytics", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("prompt_analytics", run_name="__main__")
    assert exc.value.code == 0


def test_module_entry_point_bad_argument_exit_nonzero(tmp_path, monkeypatch):
    """A bad argument surfaces a non-zero exit code through the entry point."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["prompt-analytics", "summary", "--since", "not-a-date", "--output-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("prompt_analytics", run_name="__main__")
    assert exc.value.code != 0


@pytest.fixture
def data(fake_claude):
    fake_claude.add("session_alpha.jsonl", project="alpha")
    fake_claude.add("session_beta.jsonl", project="beta")
    fake_claude.add("session_gamma.jsonl", project="gamma")
    return fake_claude


# ---------------------------------------------------------------------------
# Dispatch + exit code 0 for every analysis subcommand (on-the-fly, 7.2).
# ---------------------------------------------------------------------------


def test_summary_dispatch_without_prior_extract(data, capsys):
    assert main(_args(data, "summary")) == 0
    out = capsys.readouterr().out
    assert "Usage summary" in out
    assert "Sessions" in out


def test_by_project_dispatch(data, capsys):
    assert main(_args(data, "by-project")) == 0
    out = capsys.readouterr().out
    assert "alpha-api" in out and "beta-cli" in out


def test_by_model_dispatch(data, capsys):
    # csv format: the rich table may fold long model names at narrow widths.
    assert main(_args(data, "by-model", "--format", "csv")) == 0
    out = capsys.readouterr().out
    assert "claude-opus-4-8" in out


def test_by_token_type_dispatch_shows_context_rent(data, capsys):
    assert main(_args(data, "by-token-type")) == 0
    out = capsys.readouterr().out
    assert "Cost by token type" in out
    assert "Context rent" in out
    assert "TOTAL" in out


def test_by_category_dispatch_suggests_categorize(data, capsys):
    assert main(_args(data, "by-category")) == 0
    out = capsys.readouterr().out
    # The "(uncategorized)" bucket is asserted structurally in test_analytics;
    # here the rendered cell can wrap at 80 cols, so check stable strings only.
    assert "Cost by category" in out
    assert "categorize" in out


def test_by_output_dispatch(fake_claude, capsys):
    # session_output.jsonl writes src/parser.py (code) + tests/test_parser.py (test).
    fake_claude.add("session_output.jsonl", project="out")
    assert main(_args(fake_claude, "by-output")) == 0
    out = capsys.readouterr().out
    assert "Output composition" in out
    assert "Python" in out
    assert "Code vs tests" in out


def test_by_output_empty_history_hints(fake_claude):
    # No JSONL history at all -> the usual "no data" path (exit 1), not a crash.
    assert main(_args(fake_claude, "by-output")) == 1


def test_by_context_dispatch(fake_claude, capsys):
    fake_claude.add("session_output.jsonl", project="out")
    assert main(_args(fake_claude, "by-context")) == 0
    out = capsys.readouterr().out
    assert "Context cost over time" in out


def test_by_context_empty_history_hints(fake_claude):
    # No JSONL history at all -> the usual "no data" path (exit 1), not a crash.
    assert main(_args(fake_claude, "by-context")) == 1


def test_by_file_dispatch(fake_claude, capsys):
    # session_output.jsonl edits src/parser.py -> a per-file footprint row. csv
    # format keeps the path intact (the rich table folds it at narrow widths).
    fake_claude.add("session_output.jsonl", project="out")
    assert main(_args(fake_claude, "by-file", "--format", "csv")) == 0
    out = capsys.readouterr().out
    assert "src/parser.py" in out


def test_by_file_empty_history_hints(fake_claude):
    # No JSONL history at all -> the usual "no data" path (exit 1), not a crash.
    assert main(_args(fake_claude, "by-file")) == 1


def test_prompts_dispatch(data, capsys):
    assert main(_args(data, "prompts", "--top", "2")) == 0
    out = capsys.readouterr().out
    assert "Top 2 prompts" in out


def test_sessions_dispatch(data, capsys):
    assert main(_args(data, "sessions")) == 0
    assert "Top sessions by cost" in capsys.readouterr().out


def test_sessions_depth_dispatch(data, capsys):
    assert main(_args(data, "sessions", "--depth")) == 0
    assert "session depth" in capsys.readouterr().out


def test_compare_dispatch(data, capsys):
    assert main(_args(data, "compare", "--providers", "anthropic,copilot")) == 0
    assert "TOTAL" in capsys.readouterr().out


def test_context_dispatch(data, capsys):
    assert main(_args(data, "context")) == 0
    assert "Accumulated context" in capsys.readouterr().out


def test_ttl_dispatch(data, capsys):
    assert main(_args(data, "ttl")) == 0
    assert "TTL expiry losses" in capsys.readouterr().out


def test_compactions_dispatch(fake_claude, capsys):
    fake_claude.add("session_delta.jsonl", project="delta")
    assert main(_args(fake_claude, "compactions", "--format", "csv")) == 0
    out = capsys.readouterr().out
    assert "sess-ddd" in out  # the delta fixture contains one compaction


def test_overhead_dispatch(data, capsys):
    assert main(_args(data, "overhead", "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "Fixed session overhead" in payload["title"]
    assert payload["rows"]


def test_model_category_dispatch(data, capsys):
    assert main(_args(data, "model-category")) == 0
    assert "model x category" in capsys.readouterr().out


def test_model_category_whatif_dispatch(data, capsys):
    assert main(_args(data, "model-category", "--whatif", "claude-sonnet-4-6")) == 0
    out = capsys.readouterr().out
    assert "claude-sonnet-4-6" in out
    assert "Re-pricing all usage" in out


def test_model_category_whatif_unknown_model_exits_2(data, capsys):
    assert main(_args(data, "model-category", "--whatif", "no-such-model")) == 2
    assert "no anthropic pricing for --whatif" in capsys.readouterr().err


def test_recommend_dispatch(data, capsys):
    # Default min-prompts=50 -> no long sessions in the fixtures, but the
    # command still runs and explains itself.
    assert main(_args(data, "recommend")) == 0
    assert "Compaction recommendations" in capsys.readouterr().out


def test_burn_rate_dispatch(data, capsys):
    assert main(_args(data, "burn-rate", "--format", "csv")) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert rows and "week_of" in rows[0]


def test_break_even_dispatch(data, capsys):
    assert main(_args(data, "break-even")) == 0
    out = capsys.readouterr().out
    assert "Plan break-even" in out
    assert "Claude Max" in out
    # No quota_log.csv in the fixture sandbox -> the fallback note shows.
    assert "No quota snapshots yet" in out


def test_break_even_enriches_from_quota_log(data, capsys):
    import csv as _csv

    quota_path = data.out / "quota_log.csv"
    quota_path.parent.mkdir(parents=True, exist_ok=True)
    with quota_path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle)
        writer.writerow(["snapshot_at", "field", "utilization_pct", "resets_at"])
        writer.writerow(["2026-06-02T00:00:00Z", "seven_day", "82", ""])
    assert main(_args(data, "break-even")) == 0
    assert "seven_day 82%" in capsys.readouterr().out


def test_by_model_compact_drops_columns(data, capsys):
    # csv keeps raw column keys; the compact view exposes the cost-driver
    # subset (no input/output) so the table fits 80 columns.
    assert main(_args(data, "by-model", "--compact", "--format", "csv")) == 0
    header = csv.reader(io.StringIO(capsys.readouterr().out)).__next__()
    assert "input" not in header and "output" not in header
    assert {"model", "cache_read", "cache_write_1h", "cost_usd"} <= set(header)


def test_sessions_project_filter(data, capsys):
    assert main(_args(data, "sessions", "--project", "alpha-api", "--format", "csv")) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert rows
    assert all(row["project"] == "alpha-api" for row in rows)


def test_sessions_project_filter_unknown_exits_1(data, capsys):
    assert main(_args(data, "sessions", "--project", "ghost")) == 1
    assert "No sessions found for project 'ghost'" in capsys.readouterr().err


def test_extract_dispatch_report_and_next_steps(data, capsys):
    assert main(_args(data, "extract")) == 0
    out = capsys.readouterr().out
    assert "Extraction report" in out
    assert "Cost:" in out  # the 7.6 mini summary
    assert "Next steps:" in out
    assert (data.out / "tokens.csv").exists()


def test_run_dispatch(data, capsys):
    assert main(_args(data, "run")) == 0
    out = capsys.readouterr().out
    assert "Step 1/3" in out and "Step 3/3" in out
    assert "Next steps:" in out


def test_run_categorize_forwards_llm_flags(data, monkeypatch, capsys):
    """3.4: run --categorize --llm/--provider/--batch reach the pipeline."""
    captured = {}

    def fake_run_categorize(**kwargs):
        captured.update(kwargs)
        return 0

    from prompt_analytics import categorize

    monkeypatch.setattr(categorize, "run_categorize", fake_run_categorize)
    code = main(
        _args(
            data,
            "run",
            "--categorize",
            "--llm",
            "--batch",
            "--provider",
            "anthropic",
            "--model",
            "claude-haiku-4-5",
        )
    )
    assert code == 0
    assert captured["use_llm"] is True
    assert captured["use_batch"] is True
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-haiku-4-5"


def test_categorize_dispatch_without_prompts_csv(fake_claude, capsys):
    # No prompts.csv in the output dir: nothing could be attempted -> exit 1
    # (audit 2026-06-11 §3.4; 0 is reserved for "nothing new to classify").
    fake_claude.out.mkdir(parents=True, exist_ok=True)
    assert main(_args(fake_claude, "categorize")) == 1
    assert "No prompts file" in capsys.readouterr().err


def test_config_init_dispatch(fake_claude, capsys):
    assert main(_args(fake_claude, "config", "init")) == 0
    assert (fake_claude.out / "config.yml").exists()


def test_dashboard_dispatch(data, monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["env"] = kwargs.get("env", {})

        class Result:
            returncode = 0

        return Result()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main(_args(data, "dashboard")) == 0
    assert "streamlit" in calls["cmd"]
    assert calls["env"]["PROMPT_ANALYTICS_OUTPUT_DIR"].endswith("out")


def test_dashboard_forwards_unknown_flags_to_streamlit(data, monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd

        class Result:
            returncode = 0

        return Result()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main(_args(data, "dashboard", "--server.port", "8599")) == 0
    # Streamlit config flags are appended after the script path, untouched.
    assert calls["cmd"][-2:] == ["--server.port", "8599"]


def test_unknown_flag_on_non_dashboard_still_exits_2(data):
    # Only `dashboard` forwards extras; other commands stay strict.
    with pytest.raises(SystemExit) as exc:
        main(_args(data, "summary", "--server.port", "8599"))
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Windows redirected output: main() must force UTF-8 on stdout/stderr (M2).
# ---------------------------------------------------------------------------


def test_main_reconfigures_cp1252_streams_to_utf8(fake_claude, monkeypatch):
    """A redirected stream on Windows < 3.15 is cp1252: previews with emojis
    or arrows would crash `--format json|csv`. main() must force UTF-8."""
    out_buffer, err_buffer = io.BytesIO(), io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out_buffer, encoding="cp1252"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(err_buffer, encoding="cp1252"))

    assert main(["config", "init", "--output-dir", str(fake_claude.out)]) == 0

    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
    assert sys.stderr.encoding.lower().replace("-", "") == "utf8"
    # Would raise UnicodeEncodeError on cp1252.
    print("café → ✅")
    sys.stdout.flush()
    assert "café → ✅".encode() in out_buffer.getvalue()


def test_main_leaves_non_reconfigurable_streams_alone(fake_claude, monkeypatch):
    """A bare StringIO (no .reconfigure) must not break main()."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert main(["config", "init", "--output-dir", str(fake_claude.out)]) == 0


# ---------------------------------------------------------------------------
# Exit codes on failure.
# ---------------------------------------------------------------------------


def test_no_data_exits_1(fake_claude, capsys):
    assert main(_args(fake_claude, "summary")) == 1
    assert "No Claude Code data found" in capsys.readouterr().err


def test_unknown_provider_exits_2(data, capsys):
    assert main(_args(data, "by-project", "--provider", "nope")) == 2
    err = capsys.readouterr().err
    assert "unknown pricing provider" in err
    assert "anthropic" in err  # the known ones are listed


def test_compare_single_provider_exits_2(data, capsys):
    assert main(_args(data, "compare", "--providers", "anthropic")) == 2
    assert "at least two" in capsys.readouterr().err


def test_invalid_pricing_file_exits_2(data, tmp_path, capsys):
    bad = tmp_path / "bad.yml"
    bad.write_text("not: [valid", encoding="utf-8")
    assert main(_args(data, "summary", "--pricing", str(bad))) == 2
    assert "Error:" in capsys.readouterr().err


def test_extract_invalid_since_exits_2(data, capsys):
    assert main(_args(data, "extract", "--since", "junk")) == 2
    assert "Invalid --since" in capsys.readouterr().err


def test_extract_permission_error_exits_2_with_message(fake_claude, monkeypatch, capsys):
    """PermissionError during extract -> friendly message + exit 2 (m4, Windows Excel lockout)."""
    from prompt_analytics import extract

    def raise_perm(*a, **kw):
        exc = PermissionError(13, "Permission denied")
        exc.filename = str(fake_claude.out / "prompts.csv")
        raise exc

    monkeypatch.setattr(extract, "run_extract", raise_perm)
    assert main(_args(fake_claude, "extract")) == 2
    err = capsys.readouterr().err
    assert "close" in err.lower()
    assert "Excel" in err or "prompts.csv" in err


def test_run_permission_error_exits_2_with_message(fake_claude, monkeypatch, capsys):
    """PermissionError during run -> same friendly message (m4)."""
    from prompt_analytics import extract

    def raise_perm(*a, **kw):
        exc = PermissionError(13, "Permission denied")
        exc.filename = str(fake_claude.out / "tokens.csv")
        raise exc

    monkeypatch.setattr(extract, "run_extract", raise_perm)
    assert main(_args(fake_claude, "run")) == 2
    err = capsys.readouterr().err
    assert "close" in err.lower()


def test_sessions_depth_and_top_are_mutually_exclusive(data, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(_args(data, "sessions", "--depth", "--top", "3"))
    assert excinfo.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_export_requires_a_mode(data, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(_args(data, "export"))
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# --format on every analysis command (7.4).
# ---------------------------------------------------------------------------

ANALYSIS_COMMANDS = [
    ("summary",),
    ("by-project",),
    ("by-project", "--pareto"),
    ("by-model",),
    ("by-model", "--compact"),
    ("by-token-type",),
    ("by-category",),
    ("model-category",),
    ("model-category", "--whatif", "claude-sonnet-4-6"),
    ("burn-rate",),
    ("timeline",),
    ("timeline", "--by", "month"),
    ("prompts", "--top", "3"),
    ("sessions",),
    ("sessions", "--depth"),
    ("compare", "--providers", "anthropic,copilot"),
]


@pytest.mark.parametrize("command", ANALYSIS_COMMANDS, ids=lambda c: " ".join(c))
def test_format_json_is_parseable(data, capsys, command):
    assert main(_args(data, *command, "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"], f"no rows for {command}"
    assert "title" in payload and "notes" in payload


@pytest.mark.parametrize("command", ANALYSIS_COMMANDS, ids=lambda c: " ".join(c))
def test_format_csv_is_parseable(data, capsys, command):
    assert main(_args(data, *command, "--format", "csv")) == 0
    out = capsys.readouterr().out
    rows = list(csv.DictReader(io.StringIO(out)))
    assert rows, f"no rows for {command}"


def test_json_values_are_raw(data, capsys):
    assert main(_args(data, "by-project", "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    row = payload["rows"][0]
    assert isinstance(row["tokens"], int)
    assert isinstance(row["cost_usd"], float)


def test_by_project_cumulative_reaches_100(data, capsys):
    # The cumulative %% column is now shown by default (no --pareto needed).
    assert main(_args(data, "by-project", "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"][-1]["cumulative_pct"] == 100.0


def test_pareto_flag_still_accepted_as_noop(data, capsys):
    # Deprecated but accepted, so published `by-project --pareto` keeps working.
    assert main(_args(data, "by-project", "--pareto", "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "cumulative_pct" in payload["rows"][0]


def test_since_until_narrows_to_one_project(data, capsys):
    # Fixtures: alpha 2026-05-01, beta 2026-05-15, gamma 2026-06-01.
    assert main(_args(data, "by-project", "--since", "2026-06-01", "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["project"] for row in payload["rows"]] == ["gamma-web"]


def test_until_drops_later_projects(data, capsys):
    assert main(_args(data, "by-project", "--until", "2026-05-10", "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["project"] for row in payload["rows"]] == ["alpha-api"]


def test_since_empty_range_exits_1(data, capsys):
    assert main(_args(data, "summary", "--since", "2999-01-01")) == 1
    assert "No data in the date range" in capsys.readouterr().err


def test_until_invalid_date_exits_2(data, capsys):
    assert main(_args(data, "summary", "--until", "2026-13-40")) == 2
    assert "Invalid --until" in capsys.readouterr().err


def test_timeline_dispatch(data, capsys):
    assert main(_args(data, "timeline", "--by", "week")) == 0
    assert "Cost by week" in capsys.readouterr().out


def test_prompts_top_limits_row_count(data, capsys):
    assert main(_args(data, "prompts", "--top", "2", "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["rows"]) == 2
    costs = [row["cost_usd"] for row in payload["rows"]]
    assert costs == sorted(costs, reverse=True)


# ---------------------------------------------------------------------------
# export --flat (7.5).
# ---------------------------------------------------------------------------


def test_export_flat_writes_denormalized_csv(data, capsys):
    assert main(_args(data, "export", "--flat")) == 0
    out_path = data.out / "flat.csv"
    assert str(out_path) in capsys.readouterr().out
    with out_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5  # one per prompt across the three fixtures
    columns = set(rows[0])
    assert {"input_tokens", "output_tokens", "total_tokens"} <= columns
    assert {"cost_anthropic_usd", "cost_copilot_usd"} <= columns
    assert {"session_id", "session_start_date", "project"} <= columns


def test_export_flat_custom_out_path(data, tmp_path):
    target = tmp_path / "elsewhere" / "export.csv"
    assert main(_args(data, "export", "--flat", "--out", str(target))) == 0
    assert target.exists()


# ---------------------------------------------------------------------------
# Cache freshness through the CLI (7.2).
# ---------------------------------------------------------------------------


def test_summary_uses_fresh_csvs_after_extract(data, capsys):
    assert main(_args(data, "extract")) == 0
    capsys.readouterr()
    assert main(_args(data, "summary", "--format", "json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any("(fresh)" in note for note in payload["notes"])


def test_summary_no_cache_forces_live_parse(data, capsys):
    assert main(_args(data, "extract")) == 0
    capsys.readouterr()
    assert main(_args(data, "summary", "--format", "json", "--no-cache")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any("live parse" in note for note in payload["notes"])


# ---------------------------------------------------------------------------
# --from-csv: trust the given CSVs as-is (D1).
# ---------------------------------------------------------------------------


def test_summary_from_csv_ignores_freshness(data, capsys):
    """--from-csv must read the CSVs even when the JSONL history is newer
    (the case where a normal summary would ignore them and live-parse)."""
    import os

    assert main(_args(data, "extract")) == 0
    capsys.readouterr()
    for jsonl in data.projects.rglob("*.jsonl"):
        stat = jsonl.stat()
        os.utime(jsonl, (stat.st_atime + 60, stat.st_mtime + 60))

    assert main(["summary", "--from-csv", str(data.out), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    source_notes = [note for note in payload["notes"] if note.startswith("Source:")]
    assert source_notes and "live parse" not in source_notes[0]
    assert str(data.out) in source_notes[0]


def test_from_csv_empty_dir_exits_1(fake_claude, capsys):
    assert main(["summary", "--from-csv", str(fake_claude.home / "nope")]) == 1
    assert "No data found in the CSVs" in capsys.readouterr().err
