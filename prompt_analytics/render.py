"""Rendering of :class:`~prompt_analytics.analytics.TableResult` results.

One entry point, :func:`render`, honoring ``--format table|csv|json`` (7.4):

* ``table`` -- a rich table on stdout, notes dimmed below it.
* ``csv``   -- raw values on stdout (header = column keys), notes on stderr
  so the CSV stream stays machine-readable.
* ``json``  -- ``{"title", "rows", "notes"}`` on stdout, raw values.

Display formatting (thousands separators, ``$``, ``%``) only ever happens
here, driven by each column's ``kind``; the result rows keep raw values.
"""

from __future__ import annotations

import csv
import json
import sys
from typing import Any

from .analytics import TableResult

__all__ = ["render", "FORMATS"]

FORMATS = ("table", "csv", "json")

_NUMERIC_KINDS = frozenset({"int", "money", "pct", "x", "num", "tokens"})

# Visual styling for the `table` format (the `csv`/`json` paths stay raw).
# Color is a bonus layered on top of the structure: Rich's Console disables it
# automatically when stdout is piped/redirected or NO_COLOR is set, so these
# never leak ANSI codes into a captured stream.
_HEADER_STYLE = "bold cyan"
_TITLE_STYLE = "bold"
# Per-kind cell color: green for money (the universal terminal convention),
# yellow to make the headline token totals pop. Everything else stays default
# to avoid a rainbow.
_KIND_STYLES = {"money": "green", "tokens": "yellow"}
# A row is a summary/total row when its first label column holds one of these.
_TOTAL_MARKERS = frozenset({"TOTAL", "ALL", "OVERALL"})


def _is_total_row(row: dict[str, Any], columns: list[Any]) -> bool:
    """Whether ``row`` is a summary/total row (emphasized in the table view)."""
    for column in columns:
        if column.kind in _NUMERIC_KINDS:
            continue
        value = row.get(column.key)
        return isinstance(value, str) and value.strip().upper() in _TOTAL_MARKERS
    return False


def _abbrev_tokens(n: int) -> str:
    """Compact token count: 1,015 / 21.1M / 1.01G (the 2.8 narrow view)."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}G"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return f"{n:,}"


def _format_cell(value: Any, kind: str) -> str:
    """Format one raw value for terminal display according to a column kind."""
    if value is None or value == "":
        return "-"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "tokens":
        return _abbrev_tokens(int(value))
    if kind == "money":
        return f"${float(value):,.2f}"
    if kind == "pct":
        return f"{float(value):.1f}%"
    if kind == "x":
        return f"x{float(value):.2f}"
    if kind == "num":
        return f"{float(value):g}"
    return str(value)


def _render_table(result: TableResult) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(
        title=result.title,
        title_justify="left",
        title_style=_TITLE_STYLE,
        header_style=_HEADER_STYLE,
    )
    for column in result.columns:
        table.add_column(
            column.label,
            justify="right" if column.kind in _NUMERIC_KINDS else "left",
            overflow="fold",
            style=_KIND_STYLES.get(column.kind),
        )
    for row in result.rows:
        cells = (_format_cell(row.get(column.key), column.kind) for column in result.columns)
        table.add_row(*cells, style="bold" if _is_total_row(row, result.columns) else None)

    console = Console()
    console.print(table)
    for note in result.notes:
        style = "yellow" if note.startswith("WARNING") else "dim"
        console.print(f"  {note}", style=style, highlight=False)


def _render_csv(result: TableResult) -> None:
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow([column.key for column in result.columns])
    for row in result.rows:
        writer.writerow([row.get(column.key, "") for column in result.columns])
    for note in result.notes:
        print(note, file=sys.stderr)


def _render_json(result: TableResult) -> None:
    payload = {"title": result.title, "rows": result.rows, "notes": result.notes}
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    print()


def render(result: TableResult, output_format: str = "table") -> None:
    """Print ``result`` in the requested format.

    Args:
        result: The tabular result to print.
        output_format: One of :data:`FORMATS`.

    Raises:
        ValueError: On an unknown format name.
    """
    if output_format == "table":
        _render_table(result)
    elif output_format == "csv":
        _render_csv(result)
    elif output_format == "json":
        _render_json(result)
    else:
        raise ValueError(f"Unknown format {output_format!r}; expected one of {FORMATS}.")
