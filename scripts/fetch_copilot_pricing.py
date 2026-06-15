"""Fetch the GitHub Copilot per-model pricing grid from the official docs.

Scrapes https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
(the page the ``copilot`` pricing provider is based on -- it lists per-token
USD prices per 1 million tokens, including Claude Fable 5) and emits a
machine-readable YAML grid. Re-run it whenever GitHub updates the page;
phase 4 wires the output into the multi-provider ``pricing.yml``.

Notes:

* The **English** page is fetched by default: column labels ("Input",
  "Cache write", ...) are the parsing contract and translated pages
  (e.g. ``/fr/``) localize them.
* Models with tiered pricing (OpenAI/Google long context) produce a
  ``long_context`` sub-entry with its token threshold.
* Loud failure: if no table or no Claude model is found, the page layout
  has changed -- the script exits non-zero instead of emitting garbage.

Usage::

    uv run python scripts/fetch_copilot_pricing.py            # YAML to stdout
    uv run python scripts/fetch_copilot_pricing.py --output copilot_pricing.yml
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

DEFAULT_URL = "https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing"
USER_AGENT = "prompt-analytics-for-claude-code/0.2 (+https://github.com/romainfjgaspard/prompt-analytics-for-claude-code)"

# Page column label -> our token-type-ish key.
COLUMN_MAP = {
    "Input": "input",
    "Cached input": "cache_read",
    "Cache write": "cache_write",
    "Output": "output",
}


class _PricingTablesParser(HTMLParser):
    """Collect every <table> on the page, tagged with the nearest heading."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._section = ""
        self._in_heading = False
        self._heading_text: list[str] = []
        self._table: dict[str, Any] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_is_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("h2", "h3"):
            self._in_heading = True
            self._heading_text = []
        elif tag == "table":
            self._table = {"section": self._section, "headers": [], "rows": []}
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._cell_is_header = tag == "th"

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h2", "h3"):
            self._in_heading = False
            self._section = " ".join("".join(self._heading_text).split())
        elif tag in ("td", "th") and self._cell is not None and self._table is not None:
            text = " ".join("".join(self._cell).split())
            if self._cell_is_header:
                self._table["headers"].append(text)
            elif self._row is not None:
                self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table["rows"].append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        elif self._in_heading:
            self._heading_text.append(data)


def model_id(display_name: str) -> str:
    """``"Claude Fable 5"`` -> ``"claude-fable-5"``; ``"GPT-5.4 mini"`` -> ``"gpt-5-4-mini"``."""
    slug = re.sub(r"[.\s/]+", "-", display_name.strip().lower())
    return re.sub(r"-{2,}", "-", slug).strip("-")


def _price(cell: str) -> float | None:
    match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", cell)
    return float(match.group(1).replace(",", "")) if match else None


def parse_grid(html: str) -> dict[str, Any]:
    """Turn the page's pricing tables into {vendor: {model_id: prices}}."""
    parser = _PricingTablesParser()
    parser.feed(html)

    vendors: dict[str, dict[str, Any]] = {}
    for table in parser.tables:
        headers = table["headers"]
        if "Model" not in headers or "Input" not in headers:
            continue  # not a pricing table
        vendor = model_id(table["section"]) or "unknown"
        grid = vendors.setdefault(vendor, {})
        for cells in table["rows"]:
            if len(cells) != len(headers):
                continue
            row = dict(zip(headers, cells, strict=True))
            prices: dict[str, Any] = {}
            for label, key in COLUMN_MAP.items():
                if label in row:
                    value = _price(row[label])
                    if value is not None:
                        prices[key] = value
            if not prices:
                continue
            entry_id = model_id(row["Model"])
            tier = row.get("Tier", "Default")
            entry = grid.setdefault(
                entry_id,
                {"display_name": row["Model"], "release_status": row.get("Release status", "")},
            )
            if tier in ("Default", ""):
                entry.update(prices)
            else:
                entry["long_context"] = {
                    "threshold_input_tokens": row.get("Threshold (input tokens)", ""),
                    **prices,
                }
    return vendors


def main() -> int:
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument("--url", default=DEFAULT_URL, help="Pricing page URL (English layout)")
    arg_parser.add_argument("--output", type=Path, help="Write YAML here instead of stdout")
    args = arg_parser.parse_args()

    request = urllib.request.Request(args.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")

    vendors = parse_grid(html)

    # Canary checks: fail loudly if the page layout changed.
    total_models = sum(len(grid) for grid in vendors.values())
    claude_models = [m for m in vendors.get("anthropic", {}) if m.startswith("claude-")]
    if total_models == 0:
        print("ERROR: no pricing table parsed -- page layout changed?", file=sys.stderr)
        return 1
    if not claude_models:
        print("ERROR: no Claude model found -- page layout changed?", file=sys.stderr)
        return 1

    document = {
        "source": args.url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unit": "USD per 1 million tokens",
        "vendors": vendors,
    }
    text = (
        "# GitHub Copilot per-model pricing, scraped from the official docs.\n"
        "# Generated by scripts/fetch_copilot_pricing.py -- do not edit by hand.\n"
        + yaml.safe_dump(document, sort_keys=True, allow_unicode=True, default_flow_style=False)
    )
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output} ({total_models} models, {len(vendors)} vendors)")
    else:
        print(text)
    print(
        f"parsed {total_models} models across {len(vendors)} vendors; "
        f"Claude models: {', '.join(sorted(claude_models))}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
