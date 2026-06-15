# Test fixtures

## Standalone session logs (`session_*.jsonl`, `agent_*.jsonl`)

Hand-written JSONL exercising specific parsing edge cases (dedup across resumed
sessions, sidechains/subagents, BOM, interleaved attribution…). They are loaded
explicitly by name in `test_extract.py`.

## Versioned format fixtures (`claude-code-<version>/`)

One directory per Claude Code JSONL **format version** we have captured, e.g.
`claude-code-2.1.173/`. Following the ccusage maintainers' advice ("pin your
parsing against fixture files per version"), `test_fixtures_versioned.py` walks
every such directory and asserts the parser still reads it cleanly — this is the
canary that catches an upstream format change.

Each version directory mirrors the `~/.claude/projects/<project>/<session>.jsonl`
layout, so the directory can be passed straight to `extract.collect(claude_dir=…)`.

### Capturing a new one

These fixtures are **anonymized copies of real logs**, produced by
[`scripts/capture_fixture.py`](../../scripts/capture_fixture.py):

```bash
python scripts/capture_fixture.py ~/.claude/projects/<…>/<session>.jsonl
```

The script keeps everything the parser counts on (structure, ids, the
attribution chain, `message.usage`, `model`, `timestamp`, version, the filter
markers) and replaces all free text and paths character-for-character, so token
totals are reproducible but no real content survives. **Review the output before
committing it.**
