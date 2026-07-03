"""Long prompt/response text must not trip csv readers.

Some Claude Code prompts and responses (pasted logs, large diffs, whole files)
exceed csv's 128 KiB per-field default. Importing the package lifts that cap
process-wide, so every reader in `prompt_analytics` can read `prompts.csv` /
`prompts_text.csv` without `_csv.Error: field larger than field limit`.
"""

from __future__ import annotations

import csv
import io

_STDLIB_DEFAULT = 131072  # csv's historical per-field cap (128 KiB)


def test_package_import_lifts_csv_field_limit():
    import prompt_analytics  # noqa: F401 — imported for its import-time side effect

    assert csv.field_size_limit() > _STDLIB_DEFAULT


def test_reader_handles_field_larger_than_stdlib_default():
    import prompt_analytics  # noqa: F401 — ensure the cap is lifted in isolation

    big = "x" * (_STDLIB_DEFAULT + 100_000)
    data = f'prompt_id,prompt_text\n1,"{big}"\n'

    rows = list(csv.DictReader(io.StringIO(data)))

    assert rows[0]["prompt_text"] == big
