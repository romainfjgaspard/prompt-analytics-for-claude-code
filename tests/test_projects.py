"""Tests for project attribution (which project a session's cwd belongs to)."""

from __future__ import annotations

import pytest

from prompt_analytics.projects import ProjectResolver, project_name

REPO = "/home/u/docparser"
VAULT = r"C:\Users\u\Documents\ObsidianVault"


# ---------------------------------------------------------------------------
# project_name: the single-path fallback (no corpus to consult).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cwd", "expected"),
    [
        ("/home/u/docparser", "docparser"),
        (r"C:\Users\u\Code\lettrage-databricks", "lettrage-databricks"),
        # A worktree bills to its repository even with no corpus.
        ("/home/u/docparser/.claude/worktrees/triton-profiling-1b", "docparser"),
        ("/home/u/docparser/.claude/worktrees/triton/src/pkg", "docparser"),
        (r"C:\Users\u\Code\lettrage\.claude\worktrees\wt1", "lettrage"),
        # A worktree entered from inside another bills to the outermost repo.
        ("/home/u/a/.claude/worktrees/x/.claude/worktrees/y", "a"),
        # `.claude` alone is not a worktree marker.
        ("/home/u/repo/.claude/settings", "settings"),
        # Degenerate: the marker at the root leaves no repo component.
        ("/.claude/worktrees/x", "x"),
        ("", ""),
    ],
)
def test_project_name_falls_back_to_worktree_then_leaf(cwd, expected):
    assert project_name(cwd) == expected


# ---------------------------------------------------------------------------
# ProjectResolver: attribution using the whole corpus.
# ---------------------------------------------------------------------------


def test_subdirectory_rolls_up_to_the_repository():
    """The bug this exists for: `cd` into a subfolder invented a project."""
    resolve = ProjectResolver(anchors=[REPO, f"{REPO}/chantier-psp/nuit-2026-08-20"])
    assert resolve(f"{REPO}/chantier-psp/nuit-2026-08-20") == "docparser"
    assert resolve(f"{REPO}/triton_server") == "docparser"
    assert resolve(REPO) == "docparser"


def test_worktree_rolls_up_to_the_repository_when_it_is_known():
    resolve = ProjectResolver(anchors=[REPO, f"{REPO}/.claude/worktrees/triton-profiling"])
    assert resolve(f"{REPO}/.claude/worktrees/triton-profiling") == "docparser"
    assert resolve(f"{REPO}/.claude/worktrees/triton-profiling/scripts") == "docparser"


def test_worktree_still_rolls_up_when_the_repository_has_no_session():
    """No anchor to find: the `.claude/worktrees` marker carries it alone."""
    resolve = ProjectResolver(anchors=[f"{REPO}/.claude/worktrees/triton-profiling"])
    assert resolve(f"{REPO}/.claude/worktrees/triton-profiling") == "docparser"


def test_only_git_tracked_cwds_are_candidates():
    """A plain container directory must not swallow the repos under it.

    ``~/Documents/Code`` is a folder, not a project; it reaches the resolver
    only if a session ran there *and* it was inside a repository.
    """
    resolve = ProjectResolver(anchors=["/home/u/Code/repo-a", "/home/u/Code/repo-b"])
    assert resolve("/home/u/Code/repo-a/src") == "repo-a"
    assert resolve("/home/u/Code/repo-b/tests/unit") == "repo-b"


def test_unknown_path_keeps_its_own_name():
    resolve = ProjectResolver(anchors=[REPO])
    assert resolve("/somewhere/else/entirely") == "entirely"
    assert resolve("") == ""


def test_windows_and_posix_paths_do_not_cross_contaminate():
    resolve = ProjectResolver(anchors=[REPO, r"C:\Users\u\Code\repo"])
    assert resolve(r"C:\Users\u\Code\repo\src") == "repo"
    assert resolve(f"{REPO}/src") == "docparser"


def test_windows_anchors_match_case_insensitively():
    resolve = ProjectResolver(anchors=[r"C:\Users\u\Code\Repo"])
    assert resolve(r"c:\users\u\code\repo\src") == "Repo"


def test_split_prefix_keeps_subdirectories_separate():
    """A document tree is one repository whose folders are the real projects."""
    resolve = ProjectResolver(anchors=[VAULT], split=[VAULT])
    assert resolve(VAULT) == "ObsidianVault"
    assert resolve(rf"{VAULT}\Boost IA - Pilotage") == "Boost IA - Pilotage"
    # Deeper still: the first component under the prefix names the project.
    assert resolve(rf"{VAULT}\Boost IA - Pilotage\Board Power BI") == "Boost IA - Pilotage"


def test_split_prefix_wins_over_the_repository_roll_up():
    resolve = ProjectResolver(anchors=[VAULT, rf"{VAULT}\Perso"], split=[VAULT])
    assert resolve(rf"{VAULT}\Perso") == "Perso"


def test_split_prefix_does_not_affect_other_trees():
    resolve = ProjectResolver(anchors=[VAULT, REPO], split=[VAULT])
    assert resolve(f"{REPO}/chantier-psp/nuit") == "docparser"


def test_resolution_is_cached_per_cwd():
    resolve = ProjectResolver(anchors=[REPO])
    first = resolve(f"{REPO}/src")
    assert resolve(f"{REPO}/src") is first
