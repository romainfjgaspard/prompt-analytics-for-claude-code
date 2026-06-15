import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Keep machine-specific environment out of every test.

    ``CLAUDE_CONFIG_DIR`` must not leak in (extract/snapshot honor it), and the
    parse cache location env vars (``LOCALAPPDATA`` on Windows,
    ``XDG_CACHE_HOME`` elsewhere) are pointed into the test tmp dir so no test
    ever touches the real per-user cache.
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv("LOCALAPPDATA", str(cache_root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    """A fake ``~/.claude/projects`` tree + output dir, with helpers.

    ``Path.home()`` is monkeypatched; the parse cache is redirected by the
    autouse ``_isolated_env`` fixture (cache env vars into the sandbox).
    """
    home = tmp_path / "home"
    projects = home / ".claude" / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    out = tmp_path / "out"

    def add(fixture_filename, project="proj"):
        """Copy a committed fixture into a project directory."""
        dest = projects / project
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / fixture_filename
        shutil.copy(FIXTURES_DIR / fixture_filename, target)
        return target

    def add_subagent(fixture_filename, project, parent_session):
        """Copy a fixture into ``<project>/<parent_session>/subagents/``."""
        dest = projects / project / parent_session / "subagents"
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / fixture_filename
        shutil.copy(FIXTURES_DIR / fixture_filename, target)
        return target

    def write(filename, events, project="proj", encoding="utf-8"):
        """Write a JSONL session file from a list of event dicts."""
        dest = projects / project
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / filename
        lines = "\n".join(json.dumps(e) for e in events) + "\n"
        target.write_text(lines, encoding=encoding)
        return target

    return SimpleNamespace(
        home=home,
        projects=projects,
        out=out,
        add=add,
        add_subagent=add_subagent,
        write=write,
    )
