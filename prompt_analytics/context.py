"""Context composition metrics (Axe D): what fills the cached, re-read context.

Axe C asks what the assistant *produced*; Axe D asks what the assistant *read*
-- the content that piles up in the context window and is paid again at every
cached turn. This module classifies each piece of context content into one of
four **sources** and measures its **size** with the local tokenizer:

* ``conversation`` -- the dialogue itself (user prompts + assistant turns),
* ``file_read`` -- file contents pulled in by ``Read``/``NotebookRead`` (kept
  per language),
* ``tool_output`` -- the output of ``Bash``/``Grep``/``Glob`` and other tools,
* ``config`` -- injected setup the harness adds (skill/tool listings, file
  references, system reminders): the in-transcript proxy for the fixed prefix
  (system prompt + CLAUDE.md + MCP defs) that ``overhead`` estimates from the
  first request's API token counts.

Like Axe C it is **metrics only**: every input (file content, tool result,
attachment) is consumed transiently to count tokens; only the integer size and
its source/language label leave this module, never a byte of the content nor a
file path. Because a single local tokenizer measures every source, the *share*
of each source is an honest ratio even though the absolute counts are an
estimate (the API reports only a per-message total).
"""

from __future__ import annotations

from typing import Any

from .compose import detect_language, serialize_tool_input

__all__ = [
    "CONTEXT_SOURCES",
    "NO_LANGUAGE",
    "tool_result_source",
    "result_text",
    "assistant_tool_metas",
    "attachment_item",
]

# The four context buckets of the Axe D taxonomy (§5ter). ``conversation`` is
# summed from prompts + usage records, not emitted as a context item; the other
# three are produced by :func:`attachment_item` / tool-result classification.
CONTEXT_SOURCES = ("conversation", "file_read", "tool_output", "config")

# Tools whose result is a *file's content* read into context (language-bearing).
# Everything else (Bash, Grep, Glob, edits, ...) lands in ``tool_output``.
_FILE_READ_TOOLS = frozenset({"Read", "NotebookRead"})

# ``language`` sentinel for sources that carry no file language.
NO_LANGUAGE = "-"


def tool_result_source(tool_name: str) -> str:
    """Map an originating tool name to the source bucket of its result."""
    return "file_read" if tool_name in _FILE_READ_TOOLS else "tool_output"


def result_text(content: Any) -> str:
    """Flatten a ``tool_result`` body to text for *token counting only*.

    A tool result is either a plain string or a list of blocks; only the
    ``text`` of each block is kept. The flattened form is never persisted.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(parts)


def assistant_tool_metas(content: Any) -> dict[str, tuple[str, str]]:
    """Map each ``tool_use`` id in a message to ``(source, language)``.

    Built from the assistant's tool calls so the matching ``tool_result`` (which
    only carries the id) can be classified: ``Read``/``NotebookRead`` become
    ``file_read`` with the file's language, everything else ``tool_output``.
    """
    metas: dict[str, tuple[str, str]] = {}
    if not isinstance(content, list):
        return metas
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        tool_id = str(block.get("id") or "")
        if not tool_id:
            continue
        name = str(block.get("name") or "")
        source = tool_result_source(name)
        language = NO_LANGUAGE
        if source == "file_read":
            raw = block.get("input")
            if isinstance(raw, dict):
                path = raw.get("file_path") or raw.get("notebook_path")
                if isinstance(path, str) and path:
                    language = detect_language(path)
        metas[tool_id] = (source, language)
    return metas


def _attachment_text(attachment: dict[str, Any]) -> str:
    """Best-effort text of an attachment for token counting (never persisted).

    Collects the common content fields across attachment subtypes: an inline
    ``content``/``snippet`` (string or structured), or list fields such as
    ``addedLines`` (the deferred-tool listing). Reference-only attachments
    (a bare ``filename``/``newDate``, no body) yield ``""`` and are skipped.
    """
    for key in ("content", "snippet"):
        value = attachment.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (list, dict)) and value:
            return serialize_tool_input(value)
    added = attachment.get("addedLines")
    if isinstance(added, list) and added:
        return "\n".join(str(line) for line in added)
    return ""


def attachment_item(attachment: Any) -> tuple[str, str, str] | None:
    """Classify an attachment into ``(source, language, text)`` or ``None``.

    File-bearing attachments (a ``filename``/``displayPath`` with a body) are
    ``file_read`` content in their file's language; bodied attachments without a
    file (skill/tool listings, reminders) are ``config``. Reference-only
    attachments with no body return ``None`` (nothing enters the context).
    """
    if not isinstance(attachment, dict):
        return None
    text = _attachment_text(attachment)
    if not text:
        return None
    filename = attachment.get("filename") or attachment.get("displayPath")
    if isinstance(filename, str) and filename:
        return "file_read", detect_language(filename), text
    return "config", NO_LANGUAGE, text
