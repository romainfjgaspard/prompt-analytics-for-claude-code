"""Shared filesystem locations (Claude config dir, parse cache).

Single source of truth for the two machine-dependent roots (08 m1/m2):

* the Claude Code data directory -- ``~/.claude`` by default, overridable with
  ``CLAUDE_CONFIG_DIR`` exactly like Claude Code itself (previously honored by
  ``snapshot`` but not by ``extract``, so a relocated install produced an
  empty extraction with a working snapshot);
* the per-user cache directory -- ``%LOCALAPPDATA%`` on Windows,
  ``$XDG_CACHE_HOME`` (default ``~/.cache``) elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "claude_config_dir",
    "claude_projects_dir",
    "claude_projects_display",
    "parse_cache_dir",
]


def claude_config_dir() -> Path:
    """``$CLAUDE_CONFIG_DIR`` when set, else ``~/.claude``."""
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".claude"


def claude_projects_dir() -> Path:
    """The directory holding the JSONL session logs."""
    return claude_config_dir() / "projects"


def claude_projects_display() -> str:
    """The projects dir for user-facing messages.

    The compact ``~/.claude/projects`` form when the default applies; the
    actual path when ``CLAUDE_CONFIG_DIR`` overrides it (the honest variant).
    """
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        return str(claude_projects_dir())
    return "~/.claude/projects"


def _cache_root() -> Path:
    """Per-user cache root, idiomatic per platform (08 m1)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    base = os.environ.get("XDG_CACHE_HOME")
    return Path(base) if base else Path.home() / ".cache"


def parse_cache_dir() -> Path:
    """Where the per-file parse cache lives."""
    return _cache_root() / "prompt-analytics" / "parse"
