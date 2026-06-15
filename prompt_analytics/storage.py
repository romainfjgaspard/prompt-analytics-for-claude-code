"""Safe file writing helpers shared by all CSV/JSON producers.

Single implementation of the write-temp-then-rename pattern (R4/A4): a crash
mid-write can never leave a truncated or half-written file behind, because the
target is only ever replaced atomically via :func:`os.replace`.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "atomic_write_csv",
    "atomic_write_json",
    "append_csv",
    "escape_csv_formula",
    "FORMULA_RISK_COLUMNS",
]

# Leading characters a spreadsheet (Excel, LibreOffice, Sheets) may interpret as
# the start of a formula. A cell beginning with one of these and containing a
# crafted prompt could run on open — classic CSV formula injection (R9c).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# Only free-form, user-controlled text columns are escaped. Numeric/enum columns
# are produced by us and must stay verbatim (escaping a "-5" would corrupt it).
FORMULA_RISK_COLUMNS = frozenset({"prompt_preview", "prompt_text"})


def escape_csv_formula(value: Any) -> Any:
    """Neutralize spreadsheet formula injection by prefixing a single quote.

    Returns ``value`` unchanged unless it is a non-empty string starting with a
    formula trigger, in which case a leading ``'`` is added — the conventional
    way to force a spreadsheet to treat the cell as literal text.
    """
    if isinstance(value, str) and value and value[0] in _FORMULA_PREFIXES:
        return "'" + value
    return value


def _atomic_write(path: Path, write_body: Any) -> None:
    """Write to a temp file in ``path``'s directory, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            write_body(handle)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically (re)write a CSV file with a header line.

    Args:
        path: Target CSV file.
        columns: Column names, in order.
        rows: Row mappings; keys must match ``columns``.
    """

    risky = [c for c in columns if c in FORMULA_RISK_COLUMNS]

    def body(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        if not risky:
            writer.writerows(rows)
            return
        for row in rows:
            writer.writerow({**row, **{c: escape_csv_formula(row.get(c)) for c in risky}})

    _atomic_write(path, body)


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically (re)write a JSON file (used by the parse cache)."""

    def body(handle: Any) -> None:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))

    _atomic_write(path, body)


def append_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Append rows to a CSV, writing the header when the file is absent OR empty.

    Reserved for true time-series files (``quota_log.csv``); everything else
    is fully regenerated through :func:`atomic_write_csv`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
