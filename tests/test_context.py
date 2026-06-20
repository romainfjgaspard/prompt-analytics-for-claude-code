"""Unit tests for the Axe D context classification helpers."""

from __future__ import annotations

import random

from prompt_analytics.context import (
    NO_LANGUAGE,
    NO_PATH,
    ContextElement,
    ContextRequest,
    assistant_tool_metas,
    attachment_item,
    attribute_context_cost,
    result_text,
    split_int,
    tool_result_source,
)
from prompt_analytics.schema import UNATTRIBUTED_SOURCE


def test_tool_result_source_splits_reads_from_the_rest():
    assert tool_result_source("Read") == "file_read"
    assert tool_result_source("NotebookRead") == "file_read"
    for name in ("Bash", "Grep", "Glob", "Edit", "Write", ""):
        assert tool_result_source(name) == "tool_output"


def test_result_text_handles_both_shapes():
    assert result_text("plain output") == "plain output"
    assert result_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    # Non-text blocks are dropped; unknown shapes flatten to empty.
    assert result_text([{"type": "image", "data": "..."}]) == ""
    assert result_text(None) == ""
    assert result_text(42) == ""


def test_assistant_tool_metas_maps_id_to_source_language_and_path():
    content = [
        {"type": "text", "text": "working"},
        {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "/p/src/app.py"}},
        {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "ls"}},
        {
            "type": "tool_use",
            "id": "n1",
            "name": "NotebookRead",
            "input": {"notebook_path": "/p/x.ipynb"},
        },
        # A read without a usable path keeps the file_read source, no language/path.
        {"type": "tool_use", "id": "r2", "name": "Read", "input": {}},
        {"type": "tool_use", "name": "Grep", "input": {}},  # no id -> skipped
    ]
    metas = assistant_tool_metas(content, cwd="/p")
    # file_read carries the file's language and its project-relative path.
    assert metas["r1"] == ("file_read", "Python", "src/app.py")
    assert metas["b1"] == ("tool_output", NO_LANGUAGE, NO_PATH)
    assert metas["n1"] == ("file_read", "Jupyter Notebook", "x.ipynb")
    assert metas["r2"] == ("file_read", NO_LANGUAGE, NO_PATH)
    assert len(metas) == 4  # the id-less Grep call is not recorded
    assert assistant_tool_metas("not a list") == {}


def test_attachment_item_classifies_files_config_and_references():
    # A bodied attachment with a filename -> file_read in the file's language/path.
    src = attachment_item(
        {"type": "file", "filename": "/p/src/a.ts", "content": "export const x = 1"}, cwd="/p"
    )
    assert src == ("file_read", "TypeScript", "src/a.ts", "export const x = 1")
    # A bodied attachment without a filename -> config (no language/path).
    listing = attachment_item({"type": "skill_listing", "content": "- verify: ..."})
    assert listing == ("config", NO_LANGUAGE, NO_PATH, "- verify: ...")
    # A list field (the deferred-tool listing) becomes config text.
    delta = attachment_item({"type": "deferred_tools_delta", "addedLines": ["A", "B"]})
    assert delta == ("config", NO_LANGUAGE, NO_PATH, "A\nB")
    # Reference-only attachments (no body) and non-dicts -> nothing in context.
    assert attachment_item({"type": "opened_file_in_ide", "filename": "/p/a.py"}) is None
    assert attachment_item({"type": "task_reminder", "content": []}) is None
    assert attachment_item(None) is None


# ---------------------------------------------------------------------------
# Cost over time (D2): split_int + the attribution walk.
# ---------------------------------------------------------------------------


def test_split_int_is_exact_and_proportional():
    # Exact apportionment over many random cases: parts are >= 0 and sum to total.
    rng = random.Random(1234)
    for _ in range(2000):
        total = rng.randint(0, 50_000)
        weights = [rng.randint(0, 80) for _ in range(rng.randint(1, 7))]
        parts = split_int(total, weights)
        assert len(parts) == len(weights)
        assert all(p >= 0 for p in parts)
        assert sum(parts) == (total if sum(weights) else 0)
    # Proportional in the clean case, leftover to the largest remainder.
    assert split_int(100, [1, 1]) == [50, 50]
    assert split_int(10, [1, 2]) == [3, 7]  # 3.33/6.66 -> floors 3/6, leftover to .66
    assert split_int(5, [0, 0]) == [0, 0]  # nothing to weigh -> all zero


def _R(prompt_id, read=0, w5m=0, w1h=0, model="m", post_compact=False):
    return ContextRequest(prompt_id, model, read, w5m, w1h, post_compact)


def test_attribution_realizes_size_times_turns_of_presence():
    """An element present more turns earns proportionally more rent."""
    # Two equal-size elements, each entering with its own prompt; three turns of
    # identical cache_read. e1 (p1) is present for all 3 turns, e2 (p2) for 2.
    elements = [
        ContextElement("p1", "file_read", "Python", "a.py", 100),
        ContextElement("p2", "file_read", "Python", "a.py", 100),
    ]
    requests = [_R("p1", read=300), _R("p1", read=300), _R("p2", read=300)]
    result = attribute_context_cost(requests, elements)
    rent = result[("file_read", "Python", "a.py", "m")][0]
    # All rent lands on the one (source, language, path, model) bucket; it equals
    # the billed total exactly (size x turns is captured inside the single bucket).
    assert rent == 900
    assert sum(v[0] for v in result.values()) == 900


def test_attribution_load_is_one_off_per_entering_prompt():
    """cache_write is split across the prompt's own elements (loading), once."""
    elements = [
        ContextElement("p1", "file_read", "Python", "a.py", 300),
        ContextElement("p1", "conversation", NO_LANGUAGE, NO_PATH, 100),
    ]
    requests = [_R("p1", w5m=400, w1h=80)]
    result = attribute_context_cost(requests, elements)
    # 400 split 300:100 -> 300/100; 80 split -> 60/20.
    assert result[("file_read", "Python", "a.py", "m")][1:] == [300, 60]
    assert result[("conversation", NO_LANGUAGE, NO_PATH, "m")][1:] == [100, 20]


def test_attribution_reconciles_to_the_billed_total():
    """Every cache token is distributed: attributed + residual == the bill."""
    rng = random.Random(7)
    elements = [
        ContextElement(f"p{i}", "file_read", "Python", f"f{i}.py", rng.randint(1, 999))
        for i in range(1, 6)
    ]
    requests = [
        _R(f"p{i}", read=rng.randint(0, 9000), w5m=rng.randint(0, 4000), w1h=rng.randint(0, 1000))
        for i in range(1, 6)
    ]
    result = attribute_context_cost(requests, elements)
    assert sum(v[0] for v in result.values()) == sum(r.cache_read for r in requests)
    assert sum(v[1] for v in result.values()) == sum(r.cache_write_5m for r in requests)
    assert sum(v[2] for v in result.values()) == sum(r.cache_write_1h for r in requests)


def test_attribution_compaction_resets_the_present_set():
    """A 0->1 post_compact transition drops the prior context from the rent base."""
    elements = [
        ContextElement("p1", "file_read", "Python", "a.py", 100),
        ContextElement("c1", "conversation", NO_LANGUAGE, NO_PATH, 100),  # the summary turn
    ]
    requests = [
        _R("p1", read=100),  # pre-compaction: only p1 present
        _R("c1", read=500, post_compact=True),  # compaction: p1 dropped, c1 enters
    ]
    result = attribute_context_cost(requests, elements)
    # The 500 post-compaction read goes entirely to the summary's conversation,
    # not to the dropped Python file (whose only rent is the pre-compaction 100).
    assert result[("conversation", NO_LANGUAGE, NO_PATH, "m")][0] == 500
    assert result[("file_read", "Python", "a.py", "m")][0] == 100


def test_attribution_unattributed_when_no_element_present():
    """Cache on a turn with no measured element accrues to (unattributed)."""
    requests = [_R("ghost", read=400, w5m=50)]
    result = attribute_context_cost(requests, elements=[])
    assert result[(UNATTRIBUTED_SOURCE, NO_LANGUAGE, NO_PATH, "m")] == [400, 50, 0]


def test_attribution_keeps_models_apart():
    """Rent is attributed per request model (pricing is per model)."""
    elements = [ContextElement("p1", "file_read", "Python", "a.py", 100)]
    requests = [_R("p1", read=200, model="opus"), _R("p1", read=200, model="haiku")]
    result = attribute_context_cost(requests, elements)
    assert result[("file_read", "Python", "a.py", "opus")][0] == 200
    assert result[("file_read", "Python", "a.py", "haiku")][0] == 200
