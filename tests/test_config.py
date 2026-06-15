"""Tests for the pure config loader and explicit initialization."""

from __future__ import annotations

import pytest

from prompt_analytics.config import (
    DEFAULT_CONFIG,
    ConfigError,
    load_config,
    write_default_config,
)


def test_load_config_returns_defaults_without_creating_file(tmp_path):
    config = load_config(tmp_path)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG  # a copy, never the shared default
    # Pure read: nothing was written.
    assert list(tmp_path.iterdir()) == []


def test_load_config_merges_partial_file_over_defaults(tmp_path):
    (tmp_path / "config.yml").write_text("features:\n  categorization: false\n", encoding="utf-8")
    config = load_config(tmp_path)
    assert config["features"]["categorization"] is False
    # Missing keys fall back to defaults.
    assert config["features"]["prompt_text"] is True


def test_load_config_invalid_yaml_raises_clear_error(tmp_path):
    (tmp_path / "config.yml").write_text("features: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="config init"):
        load_config(tmp_path)


def test_write_default_config_creates_and_respects_existing(tmp_path):
    path = write_default_config(tmp_path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")

    path.write_text("features:\n  prompt_text: false\n", encoding="utf-8")
    write_default_config(tmp_path)  # no overwrite by default
    assert load_config(tmp_path)["features"]["prompt_text"] is False

    write_default_config(tmp_path, overwrite=True)
    assert path.read_text(encoding="utf-8") == content
