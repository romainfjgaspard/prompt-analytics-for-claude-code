"""prompt-analytics-for-claude-code: extract and analyze Claude Code usage from local JSONL files."""

import csv as _csv
import sys as _sys

__version__ = "0.5.0"

__all__ = ["__version__"]


def _raise_csv_field_limit() -> None:
    """Lift csv's per-field size cap, once and process-wide.

    Some Claude Code prompts and responses (pasted logs, large diffs, whole
    files) exceed the 128 KiB stdlib default, which makes ``csv.reader`` raise
    ``_csv.Error: field larger than field limit`` when reading ``prompts.csv``
    / ``prompts_text.csv``. ``field_size_limit`` is a module-global setting, so
    a single call on import covers every reader in the package. Step down from
    ``sys.maxsize`` until the C long accepts it (portable across 32/64-bit).
    """
    limit = _sys.maxsize
    while True:
        try:
            _csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()
