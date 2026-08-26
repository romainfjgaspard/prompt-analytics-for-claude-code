"""Project attribution: which project a session's working directory belongs to.

Claude Code records a session's ``cwd`` and nothing else about the workspace --
no repository root, no project id. Taking the final path component of that cwd
(the original rule) makes every subdirectory you happen to ``cd`` into look like
a separate project: one repository showed up as ``docparser``,
``triton-profiling-1b``, ``nuit-2026-08-20`` and ``triton_server``, its cost
split four ways.

The rule here rolls a cwd up to the directory that actually is the project:

1. an explicit **split prefix** wins (see below);
2. otherwise the **shallowest ancestor that is itself a session cwd inside a git
   repository** -- a data-driven stand-in for "the repository root", since the
   real one cannot be probed: the logs may come from another machine or another
   OS, and the same CSVs must attribute identically wherever they are read;
3. otherwise the ``.claude/worktrees`` marker, for a worktree whose parent
   repository never hosted a session of its own;
4. otherwise the final component, the historical behaviour.

**Split prefixes** exist because "one repository, one project" is right for code
and wrong for a document tree. An Obsidian vault is a single git repository
whose top-level folders are the real units of work; rolling them up collapses
twenty-one of them into one line. No signal in the data separates that from a
source tree, so it is a declaration, not a deduction: list such a tree under
``projects.split`` in ``config.yml`` and its immediate subdirectories stay
separate projects.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePath, PurePosixPath, PureWindowsPath

__all__ = ["ProjectResolver", "project_name", "pure_path"]

# Claude Code checks git worktrees out under `<repo>/.claude/worktrees/<name>`.
_WORKTREE_MARKER = (".claude", "worktrees")


def pure_path(path: str) -> PurePath:
    """Parse ``path`` with the flavour it was written in.

    Windows paths are recognized by their backslashes, so a log written on
    Windows parses correctly when read from Linux/WSL and vice versa. Comparison
    then follows that flavour's rules -- case-insensitive for Windows.
    """
    return PureWindowsPath(path) if "\\" in path else PurePosixPath(path)


def _worktree_repo(path: PurePath) -> str | None:
    """The repository owning ``path`` when it sits inside a git worktree.

    Anchors on the ``.claude/worktrees`` pair rather than a fixed depth (the
    session may sit deeper inside the worktree), and scans left to right, so a
    worktree entered from inside another still bills to the outermost repo.
    """
    parts = path.parts
    for index in range(2, len(parts)):
        if (parts[index - 1], parts[index]) == _WORKTREE_MARKER:
            repo = parts[index - 2]
            # Guard `/.claude/worktrees/x`, where the candidate is the path
            # anchor (`/` or `C:\`) rather than a directory.
            if repo != path.anchor:
                return repo
    return None


def project_name(cwd: str) -> str:
    """The project for ``cwd`` with no dataset to consult: worktree, else leaf.

    The single-path fallback used by :class:`ProjectResolver` and by callers
    that have one path and no corpus (a snapshot, a test).
    """
    if not cwd:
        return ""
    path = pure_path(cwd)
    return _worktree_repo(path) or path.name


class ProjectResolver:
    """Maps a session ``cwd`` to a project name, using the whole dataset.

    Attribution is not a per-row decision: knowing that ``…/docparser`` is a
    project and ``…/docparser/chantier-psp/nuit-2026-08-20`` is a folder inside
    it requires seeing both. Build one resolver from every cwd in the corpus,
    then call it per row.

    Args:
        anchors: The cwds that sit inside a git repository -- candidates for
            "this is the project root". Passing a cwd with no git branch would
            let a plain container directory (``~/Documents/Code``) swallow every
            repository under it.
        split: Path prefixes whose immediate subdirectories stay separate
            projects (``projects.split`` in ``config.yml``).
    """

    def __init__(self, anchors: Iterable[str], split: Iterable[str] = ()) -> None:
        # Path -> the spelling to report. Windows paths compare case-insensitively,
        # so the same repository reached as `C:\Code\Repo` and `c:\code\repo` must
        # yield ONE project row; sorting makes the winning spelling deterministic
        # rather than dependent on file-iteration order.
        self._anchors: dict[PurePath, str] = {}
        for anchor in sorted(a for a in anchors if a):
            self._anchors.setdefault(pure_path(anchor), pure_path(anchor).name)
        self._split = [pure_path(s) for s in split if s]
        self._cache: dict[str, str] = {}

    def __call__(self, cwd: str) -> str:
        """The project name for ``cwd`` (``""`` for an empty cwd)."""
        if not cwd:
            return ""
        cached = self._cache.get(cwd)
        if cached is None:
            cached = self._resolve(pure_path(cwd))
            self._cache[cwd] = cached
        return cached

    def _resolve(self, path: PurePath) -> str:
        for prefix in self._split:
            below = _relative_parts(path, prefix)
            if below is None:
                continue
            # The prefix itself keeps its own name; anything below it is named
            # after its first component under the prefix.
            return below[0] if below else path.name

        # Shallowest first, so the repository wins over a worktree or a nested
        # checkout inside it. Strict ancestors only: a cwd that is its own
        # shallowest anchor carries no information beyond its name, and letting
        # it match here would shadow the worktree rule below.
        for candidate in reversed(path.parents):
            name = self._anchors.get(candidate)
            if name:
                return name

        return project_name(str(path))


def _relative_parts(path: PurePath, prefix: PurePath) -> tuple[str, ...] | None:
    """``path``'s components below ``prefix``, or ``None`` if not below it.

    An empty tuple means ``path`` *is* ``prefix``. Paths of different flavours
    never match -- they come from different machines.
    """
    if type(path) is not type(prefix):
        return None
    try:
        if not path.is_relative_to(prefix):
            return None
    except ValueError:  # pragma: no cover - defensive, is_relative_to is total
        return None
    return path.relative_to(prefix).parts
