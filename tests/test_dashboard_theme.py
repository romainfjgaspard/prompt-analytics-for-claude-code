"""The ``dashboard`` command forces the branded dark theme for installed launches.

Streamlit otherwise follows the visitor's OS theme and only reads
``.streamlit/config.toml`` from the launch cwd -- which an installed tool
(``uv tool`` / ``pip``) never has, so the board would render light against the
dark-only ECharts chrome. These tests pin the forced theme and guard against
drift between it and the ``.streamlit/config.toml`` used by the deployed demo.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import prompt_analytics.cli as cli
from prompt_analytics.cli import DASHBOARD_THEME_ENV

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TOML = REPO_ROOT / ".streamlit" / "config.toml"

# config.toml [theme] key -> value that the forced env theme must reproduce.
_EXPECTED_THEME = {
    "base": "dark",
    "primaryColor": "#D97757",
    "backgroundColor": "#0B1220",
    "secondaryBackgroundColor": "#111827",
    "textColor": "#F8FAFC",
    "borderColor": "#2B3954",
}


def test_dashboard_theme_forces_dark() -> None:
    assert DASHBOARD_THEME_ENV["STREAMLIT_THEME_BASE"] == "dark"


def test_dashboard_theme_values_match_config_toml() -> None:
    """The forced env theme must equal the deployed demo's config.toml [theme]."""
    cfg = CONFIG_TOML.read_text(encoding="utf-8")
    forced_values = set(DASHBOARD_THEME_ENV.values())
    for key, value in _EXPECTED_THEME.items():
        assert f'{key} = "{value}"' in cfg, f"{key} drifted from .streamlit/config.toml"
        assert value in forced_values, f"{key} not forced by DASHBOARD_THEME_ENV"


def test_handle_dashboard_injects_theme_env(monkeypatch, tmp_path) -> None:
    """``dashboard`` passes STREAMLIT_THEME_* to the streamlit subprocess."""
    captured: dict[str, dict[str, str]] = {}

    def fake_run(cmd, env=None, check=False):  # noqa: ANN001, ARG001
        captured["env"] = env or {}

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = argparse.Namespace(output_dir=str(tmp_path), no_refresh=True, streamlit_args=[])
    assert cli._handle_dashboard(args) == 0
    assert captured["env"]["STREAMLIT_THEME_BASE"] == "dark"
    assert captured["env"]["STREAMLIT_THEME_PRIMARY_COLOR"] == "#D97757"


def test_handle_dashboard_respects_user_theme_override(monkeypatch, tmp_path) -> None:
    """An explicit user STREAMLIT_THEME_* wins over the forced default."""
    captured: dict[str, dict[str, str]] = {}

    def fake_run(cmd, env=None, check=False):  # noqa: ANN001, ARG001
        captured["env"] = env or {}

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("STREAMLIT_THEME_BASE", "light")

    args = argparse.Namespace(output_dir=str(tmp_path), no_refresh=True, streamlit_args=[])
    assert cli._handle_dashboard(args) == 0
    assert captured["env"]["STREAMLIT_THEME_BASE"] == "light"
