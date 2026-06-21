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

from collections import defaultdict
from typing import Any, NamedTuple

from .compose import detect_language, relativize, serialize_tool_input
from .schema import UNATTRIBUTED_SOURCE

__all__ = [
    "CONTEXT_SOURCES",
    "NO_LANGUAGE",
    "NO_PATH",
    "tool_result_source",
    "result_text",
    "assistant_tool_metas",
    "attachment_item",
    "ContextElement",
    "ContextRequest",
    "split_int",
    "attribute_context_cost",
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

# ``path`` sentinel for sources that are not a single file (conversation, tool
# output, config) -- only ``file_read`` carries a project-relative file path.
NO_PATH = "-"


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


def assistant_tool_metas(content: Any, cwd: str = "") -> dict[str, tuple[str, str, str]]:
    """Map each ``tool_use`` id in a message to ``(source, language, path)``.

    Built from the assistant's tool calls so the matching ``tool_result`` (which
    only carries the id) can be classified: ``Read``/``NotebookRead`` become
    ``file_read`` with the file's language and its project-relative path (the
    identity the unified per-file view joins on), everything else ``tool_output``
    with no language/path.
    """
    metas: dict[str, tuple[str, str, str]] = {}
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
        path = NO_PATH
        if source == "file_read":
            raw = block.get("input")
            if isinstance(raw, dict):
                raw_path = raw.get("file_path") or raw.get("notebook_path")
                if isinstance(raw_path, str) and raw_path:
                    language = detect_language(raw_path)
                    path = relativize(raw_path, cwd)
        metas[tool_id] = (source, language, path)
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


def attachment_item(attachment: Any, cwd: str = "") -> tuple[str, str, str, str] | None:
    """Classify an attachment into ``(source, language, path, text)`` or ``None``.

    File-bearing attachments (a ``filename``/``displayPath`` with a body) are
    ``file_read`` content in their file's language, carrying its project-relative
    path; bodied attachments without a file (skill/tool listings, reminders) are
    ``config`` with no language/path. Reference-only attachments with no body
    return ``None`` (nothing enters the context).
    """
    if not isinstance(attachment, dict):
        return None
    text = _attachment_text(attachment)
    if not text:
        return None
    filename = attachment.get("filename") or attachment.get("displayPath")
    if isinstance(filename, str) and filename:
        return "file_read", detect_language(filename), relativize(filename, cwd), text
    return "config", NO_LANGUAGE, NO_PATH, text


# ---------------------------------------------------------------------------
# Cost over time (Axe D, D2): the real cost of a context element.
#
# The static snapshot above answers *what* fills the context; this answers
# *what it costs*. A context element's real cost is its size times the number
# of turns it stays in context (it is re-read -- billed as ``cache_read`` --
# every single turn). We reconstruct membership from the ``parentUuid`` chain
# (a piece entering with a prompt stays until a compaction drops it), then
# attribute the **real** per-request cache tokens across the elements present
# that turn, proportionally to their measured size:
#
# * **rent** -- each request's billed ``cache_read`` is split across the
#   elements present (entered at or before this turn, since the last
#   compaction). Summed over turns this realizes "size x turns of presence":
#   a big file kept 80 turns earns ~80x the rent of one read once.
# * **load** -- each request's billed ``cache_write`` is split across the
#   elements *entering* with its prompt: the one-off cost of caching them.
#
# The split is exact (largest-remainder integer apportionment), so every
# request's cache tokens are fully distributed; whatever lands on a turn with
# no measured element (chiefly the synthetic post-compaction summary, which we
# never persist as content) accrues to the ``(unattributed)`` source. Hence the
# attributed totals reconcile to the billed ``cache_read``/``cache_write`` of
# the main chain *to the token* -- the rigour signature -- while the per-element
# share is the same honest proportional estimate as the prose/code split.
# ---------------------------------------------------------------------------


class ContextElement(NamedTuple):
    """One measured piece of context, tagged with the prompt it enters with.

    ``prompt_id`` places the element on the ``parentUuid`` chain (it enters when
    that prompt's first main-chain request runs); ``(source, language, path)`` is
    the aggregation key (``path`` is the project-relative file for ``file_read``,
    ``NO_PATH`` otherwise); ``tokens`` is its local-tokenizer size. Metrics only.
    """

    prompt_id: str
    source: str
    language: str
    path: str
    tokens: int


class ContextRequest(NamedTuple):
    """One main-chain API request, where the cache is actually billed.

    Chronological order across a session is the caller's responsibility.
    ``post_compact`` marks requests answering the synthetic post-compaction
    continuation: a 0->1 transition is a compaction event that drops the prior
    context (the present set resets).
    """

    prompt_id: str
    model: str
    cache_read: int
    cache_write_5m: int
    cache_write_1h: int
    post_compact: bool


def split_int(total: int, weights: list[int]) -> list[int]:
    """Apportion ``total`` across ``weights`` so the parts sum to ``total`` exactly.

    Largest-remainder method: floor each share, then hand the leftover units to
    the largest fractional remainders (ties broken by position, for
    determinism). ``weights`` must be non-negative with a positive sum; returns
    a list the same length as ``weights``. This is what makes the attribution
    reconcile to the token -- no rounding ever creates or loses a cache token.
    """
    total_weight = sum(weights)
    if total_weight <= 0:
        return [0] * len(weights)
    scaled = [total * w for w in weights]
    floors = [s // total_weight for s in scaled]
    leftover = total - sum(floors)
    if leftover:
        # Distribute the remaining units to the largest remainders.
        order = sorted(
            range(len(weights)),
            key=lambda i: (scaled[i] - floors[i] * total_weight, -i),
            reverse=True,
        )
        for i in order[:leftover]:
            floors[i] += 1
    return floors


# Accumulator value: [rent_read, load_write_5m, load_write_1h].
_CostTriple = list[int]


def attribute_context_cost(
    requests: list[ContextRequest],
    elements: list[ContextElement],
) -> dict[tuple[str, str, str, str], _CostTriple]:
    """Attribute one session's real cache tokens across its context elements.

    Walks ``requests`` in the given (chronological) order, maintaining the set
    of elements present in context (reset at every compaction event), and splits
    each request's billed ``cache_read`` (rent) and ``cache_write`` (load) by
    measured size (see the module note above).

    Returns:
        ``(source, language, path, model) -> [rent_read, load_5m, load_1h]`` in
        tokens. Cache that cannot be tied to a measured element is keyed under
        ``(UNATTRIBUTED_SOURCE, NO_LANGUAGE, NO_PATH, model)``. Summing every
        value's first column equals the total billed ``cache_read`` of
        ``requests`` exactly; the load columns likewise total ``cache_write``.
    """
    elements_by_prompt: dict[str, list[ContextElement]] = defaultdict(list)
    for element in elements:
        if element.tokens > 0:
            elements_by_prompt[element.prompt_id].append(element)

    result: dict[tuple[str, str, str, str], _CostTriple] = defaultdict(lambda: [0, 0, 0])
    # Present context as (source, language, path) -> tokens, rebuilt as prompts
    # enter and cleared at each compaction. ``entered`` guards against re-adding a
    # prompt's elements (its requests span several turns).
    present: dict[tuple[str, str, str], int] = defaultdict(int)
    entered: set[str] = set()
    prev_post_compact = False

    for req in requests:
        if req.post_compact and not prev_post_compact:
            present.clear()  # a compaction drops the prior context
        prev_post_compact = req.post_compact

        prompt_elements = elements_by_prompt.get(req.prompt_id, [])
        if req.prompt_id not in entered:
            entered.add(req.prompt_id)
            for element in prompt_elements:
                present[(element.source, element.language, element.path)] += element.tokens

        # Rent: split this turn's cache_read across everything currently present.
        if req.cache_read:
            keys = list(present)
            shares = split_int(req.cache_read, [present[k] for k in keys])
            if shares and sum(shares):
                for key, share in zip(keys, shares, strict=True):
                    result[(key[0], key[1], key[2], req.model)][0] += share
            else:
                result[(UNATTRIBUTED_SOURCE, NO_LANGUAGE, NO_PATH, req.model)][0] += req.cache_read

        # Load: split this turn's cache_write across the prompt's own elements.
        for ttl_index, write in ((1, req.cache_write_5m), (2, req.cache_write_1h)):
            if not write:
                continue
            shares = split_int(write, [element.tokens for element in prompt_elements])
            if shares and sum(shares):
                for element, share in zip(prompt_elements, shares, strict=True):
                    result[(element.source, element.language, element.path, req.model)][
                        ttl_index
                    ] += share
            else:
                result[(UNATTRIBUTED_SOURCE, NO_LANGUAGE, NO_PATH, req.model)][ttl_index] += write

    return result
