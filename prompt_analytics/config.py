"""Configuration loading and environment handling.

:func:`load_config` is a pure read: it never creates files. The default
``config.yml`` is written explicitly via :func:`write_default_config`
(exposed as the ``config init`` CLI subcommand).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DEFAULT_CONFIG", "ConfigError", "load_config", "write_default_config"]

DEFAULT_CONFIG: dict[str, Any] = {
    "features": {"categorization": True, "prompt_text": True, "quota_snapshot": True},
}

CONFIG_FILENAME = "config.yml"


class ConfigError(Exception):
    """Raised when ``config.yml`` exists but cannot be parsed."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``.

    Nested dictionaries are merged key by key; any non-dict value in ``override``
    replaces the corresponding value in ``base``. ``base`` is mutated in place and
    also returned for convenience.

    Args:
        base: The mapping to merge into (typically a copy of the defaults).
        override: The mapping whose values take precedence.

    Returns:
        The merged mapping (the same object as ``base``).
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(output_dir: Path) -> dict[str, Any]:
    """Load runtime configuration, falling back to defaults. Read-only.

    Reads ``<output_dir>/config.yml`` and deep-merges it over
    :data:`DEFAULT_CONFIG` so any missing key falls back to its default. If the
    file does not exist, a copy of the defaults is returned -- nothing is
    written to disk (use :func:`write_default_config` for that).

    Args:
        output_dir: Directory that may contain ``config.yml``.

    Returns:
        A complete configuration mapping (never missing top-level keys).

    Raises:
        ConfigError: If ``config.yml`` exists but is not valid YAML.
    """
    config_path = output_dir / CONFIG_FILENAME
    merged = copy.deepcopy(DEFAULT_CONFIG)

    if not config_path.exists():
        return merged

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML in {config_path}: {exc}\n"
            "Fix the file or delete it and run `prompt-analytics config init`."
        ) from exc

    if isinstance(loaded, dict):
        _deep_merge(merged, loaded)
    return merged


def write_default_config(output_dir: Path, *, overwrite: bool = False) -> Path:
    """Write the default ``config.yml`` into ``output_dir``.

    Args:
        output_dir: Target directory (created if missing).
        overwrite: Replace an existing file when True; otherwise an existing
            file is left untouched.

    Returns:
        The path of the config file (written or pre-existing).
    """
    config_path = output_dir / CONFIG_FILENAME
    if config_path.exists() and not overwrite:
        return config_path
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_CONFIG, handle, default_flow_style=False, sort_keys=False)
    return config_path
