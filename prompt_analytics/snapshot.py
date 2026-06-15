"""Snapshot Claude quota utilization via the undocumented OAuth usage endpoint.

WARNING — read before using:

1. **Token reuse**: This command reads ``~/.claude/.credentials.json`` and
   reuses the OAuth session token that Claude Code uses to authenticate with
   Anthropic.  That token grants access to your Anthropic account; treat it
   like a password.  Never share the token or commit it to version control.

2. **Undocumented endpoint**: ``API_URL`` is not part of any public Anthropic
   API.  It was discovered via reverse-engineering by the open-source project
   usage-monitor-for-claude (https://github.com/jens-duttke/usage-monitor-for-claude).
   Anthropic may change or remove it at any time without notice.

3. **Honest User-Agent**: This tool identifies itself as
   ``prompt-analytics-for-claude-code/<version>`` — it does NOT impersonate
   the Claude Code client.

On any error (missing credentials, HTTP error, network timeout) this module
prints a clear warning to stderr and returns 0 without raising, so callers
are never blocked.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__, paths, storage
from .schema import QUOTA_LOG_COLS

__all__ = ["run_snapshot", "parse_rows", "append_csv"]

# Undocumented Anthropic internal usage endpoint.
# Source: https://github.com/jens-duttke/usage-monitor-for-claude
# Monitor that project for changes if this URL breaks.
API_URL = "https://api.anthropic.com/api/oauth/usage"

# Beta header required by the endpoint (follow usage-monitor-for-claude for updates).
BETA_HEADER = "oauth-2025-04-20"

CSV_COLUMNS = QUOTA_LOG_COLS


def _credentials_path() -> Path:
    """Return the path to the Claude credentials file (resolved at call time)."""
    return paths.claude_config_dir() / ".credentials.json"


def read_access_token() -> str | None:
    """Read the Claude OAuth access token from the local credentials file.

    Returns:
        The access token, or ``None`` if the file is absent, unreadable, or
        does not contain a token.
    """
    path = _credentials_path()
    if not path.exists():
        return None
    try:
        creds = json.loads(path.read_text(encoding="utf-8"))
        token = creds.get("claudeAiOauth", {}).get("accessToken")
        return token or None
    except Exception:
        return None


def fetch_usage(token: str) -> dict[str, Any]:
    """Fetch the current quota usage from the OAuth usage endpoint.

    Args:
        token: The OAuth access token from ``~/.claude/.credentials.json``.

    Returns:
        The parsed JSON response.

    Raises:
        urllib.error.HTTPError: On any non-2xx HTTP response (including 401).
        urllib.error.URLError: On network errors or timeouts.
    """
    req = urllib.request.Request(
        API_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": f"prompt-analytics-for-claude-code/{__version__}",
            "anthropic-beta": BETA_HEADER,
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    return body


def parse_rows(data: dict[str, Any], snapshot_at: str) -> list[dict[str, Any]]:
    """Turn the API response into one CSV row per quota field.

    Args:
        data: The parsed API response.
        snapshot_at: ISO timestamp recorded for every row.

    Returns:
        A list of row dictionaries matching :data:`CSV_COLUMNS`.
    """
    rows: list[dict[str, Any]] = []
    for field, value in data.items():
        if field == "extra_usage":
            continue
        if not isinstance(value, dict) or "utilization" not in value:
            continue
        utilization = value.get("utilization")
        if utilization is None:
            continue
        rows.append(
            {
                "snapshot_at": snapshot_at,
                "field": field,
                "utilization_pct": round(float(utilization), 2),
                "resets_at": value.get("resets_at", ""),
            }
        )
    return rows


def append_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Append rows to ``quota_log.csv`` via the shared storage helper.

    The header is written when the file is absent or empty.  ``quota_log.csv``
    is the only append-only output (a time series); everything else is
    regenerated atomically.

    Args:
        rows: The rows to append.
        path: Target CSV file path.
    """
    storage.append_csv(path, CSV_COLUMNS, rows)


def run_snapshot(output_dir: Path) -> int:
    """Snapshot Claude quota utilization into ``<output_dir>/quota_log.csv``.

    Appends one row per quota field (columns: snapshot_at, field,
    utilization_pct, resets_at).  Returns 0 on any failure without raising.

    Args:
        output_dir: Directory where ``quota_log.csv`` is written.

    Returns:
        Number of rows appended (0 on any failure).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quota_log.csv"
    snapshot_at = datetime.now(timezone.utc).isoformat()

    try:
        token = read_access_token()
        if not token:
            print(
                f"Warning: no Claude access token found at {_credentials_path()}; "
                "skipping quota snapshot.",
                file=sys.stderr,
            )
            return 0

        data = fetch_usage(token)
        rows = parse_rows(data, snapshot_at)
        if not rows:
            print(
                "Warning: no quota fields found in the usage response; skipping quota snapshot.",
                file=sys.stderr,
            )
            return 0

        append_csv(rows, csv_path)

    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print(
                "Warning: quota snapshot failed: 401 Unauthorized — "
                "your Claude OAuth token has expired. "
                "Restart Claude Code to refresh it.",
                file=sys.stderr,
            )
        else:
            print(
                f"Warning: quota snapshot failed: HTTP {exc.code} {exc.reason}",
                file=sys.stderr,
            )
        return 0

    except Exception as exc:  # noqa: BLE001 — graceful failure, never raise.
        print(f"Warning: quota snapshot failed: {exc}", file=sys.stderr)
        return 0

    for row in rows:
        print(f"  {row['field']}: {row['utilization_pct']}% (resets {row['resets_at']})")
    print(f"Appended {len(rows)} row(s) to {csv_path}")
    return len(rows)
