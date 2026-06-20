"""Unit tests for the Axe D context classification helpers."""

from __future__ import annotations

from prompt_analytics.context import (
    NO_LANGUAGE,
    assistant_tool_metas,
    attachment_item,
    result_text,
    tool_result_source,
)


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


def test_assistant_tool_metas_maps_id_to_source_and_language():
    content = [
        {"type": "text", "text": "working"},
        {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "/p/app.py"}},
        {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "ls"}},
        {
            "type": "tool_use",
            "id": "n1",
            "name": "NotebookRead",
            "input": {"notebook_path": "/p/x.ipynb"},
        },
        # A read without a usable path keeps the file_read source, no language.
        {"type": "tool_use", "id": "r2", "name": "Read", "input": {}},
        {"type": "tool_use", "name": "Grep", "input": {}},  # no id -> skipped
    ]
    metas = assistant_tool_metas(content)
    assert metas["r1"] == ("file_read", "Python")
    assert metas["b1"] == ("tool_output", NO_LANGUAGE)
    assert metas["n1"] == ("file_read", "Jupyter Notebook")
    assert metas["r2"] == ("file_read", NO_LANGUAGE)
    assert len(metas) == 4  # the id-less Grep call is not recorded
    assert assistant_tool_metas("not a list") == {}


def test_attachment_item_classifies_files_config_and_references():
    # A bodied attachment with a filename -> file_read in the file's language.
    src = attachment_item({"type": "file", "filename": "/p/a.ts", "content": "export const x = 1"})
    assert src == ("file_read", "TypeScript", "export const x = 1")
    # A bodied attachment without a filename -> config.
    listing = attachment_item({"type": "skill_listing", "content": "- verify: ..."})
    assert listing == ("config", NO_LANGUAGE, "- verify: ...")
    # A list field (the deferred-tool listing) becomes config text.
    delta = attachment_item({"type": "deferred_tools_delta", "addedLines": ["A", "B"]})
    assert delta == ("config", NO_LANGUAGE, "A\nB")
    # Reference-only attachments (no body) and non-dicts -> nothing in context.
    assert attachment_item({"type": "opened_file_in_ide", "filename": "/p/a.py"}) is None
    assert attachment_item({"type": "task_reminder", "content": []}) is None
    assert attachment_item(None) is None
