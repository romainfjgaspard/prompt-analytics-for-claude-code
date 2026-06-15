"""Reconcile prompt-analytics token totals against ccusage (milestone J2).

Compares, day by day and model by model, the token totals produced by our
parser (global dedup, latest usage snapshot per message, all files including
subagents) with the output of ``bunx ccusage daily --json`` on the real local
history (``~/.claude/projects``). Every discrepancy must be zero or explained
by a known, documented difference; anything else fails the run (exit code 1).

Methodology:

* The whole ``projects`` tree is first copied to a temp directory and BOTH
  tools read the frozen copy (ours directly, ccusage via the
  ``CLAUDE_CONFIG_DIR`` environment variable) -- a live Claude Code session
  appends to the logs continuously, so comparing two reads of the live tree
  can never converge.
* Daily buckets use the same explicit timezone on both sides.

Known, accepted differences (annotated line by line in the output):

* ``<synthetic>`` -- Claude Code writes synthetic (non-API) assistant
  messages with model ``<synthetic>``; ccusage skips them entirely, we list
  them (they normally carry zero usage).
* ``--timezone`` other than UTC -- ccusage v20 accepts the flag but ignores
  it for daily bucketing (verified empirically: an event at 22:30Z stays on
  the UTC date under ``--timezone Europe/Paris``). Reconcile in UTC only;
  day-boundary diffs under any other timezone are a ccusage limitation.

Usage::

    uv run python scripts/reconcile_ccusage.py [--timezone UTC] [--keep-json]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_analytics.extract import _iter_jsonl_files, parse_file  # noqa: E402
from prompt_analytics.schema import UsageRecord  # noqa: E402

# (our token_type(s), ccusage field) pairs compared per (date, model).
FIELDS = [
    (("input",), "inputTokens"),
    (("output",), "outputTokens"),
    (("cache_read",), "cacheReadTokens"),
    (("cache_write_5m", "cache_write_1h"), "cacheCreationTokens"),
]

SYNTHETIC = "<synthetic>"


def collect_ours(projects_dir: Path, tz: ZoneInfo) -> dict[tuple[str, str], Counter[str]]:
    """Aggregate our parser's deduplicated usage by (local date, model).

    Same counting rules as ``run_extract``: global dedup by
    ``message.id + requestId``, keeping the LARGEST usage snapshot per key
    (message lines carry progressive usage; ties keep the first line, as a
    message can straddle midnight).
    """
    chosen: dict[str, UsageRecord] = {}
    keyless: list[UsageRecord] = []
    files = _iter_jsonl_files(projects_dir)
    skipped = 0

    def magnitude(record: UsageRecord) -> int:
        return sum(record["tokens"].values())

    for filepath in files:
        try:
            parsed = parse_file(filepath)
        except (OSError, UnicodeDecodeError) as exc:
            print(f"note: skipped unreadable {filepath}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        for record in parsed["usage"]:
            key = record["dedup_key"]
            if not key:
                keyless.append(record)
                continue
            previous = chosen.get(key)
            if previous is None or magnitude(record) > magnitude(previous):
                chosen[key] = record

    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in [*chosen.values(), *keyless]:
        try:
            stamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        day = stamp.astimezone(tz).date().isoformat()
        totals[(day, record["model"] or "(unknown)")].update(record["tokens"])
    print(f"ours: {len(files) - skipped} files parsed, {len(chosen)} unique usage keys")
    return totals


def collect_ccusage(config_dir: Path, tz_name: str) -> dict[tuple[str, str], Counter[str]]:
    """Aggregate ccusage's daily JSON by (date, model), on the frozen copy."""
    if shutil.which("bunx"):
        runner = "bunx"
    elif shutil.which("npx"):
        runner = "npx"
    else:
        raise SystemExit("Neither bunx nor npx found; install bun or node.")
    cmd = f"{runner} ccusage daily --json --offline --timezone {tz_name}"
    print(f"running: {cmd} (CLAUDE_CONFIG_DIR={config_dir})")
    env = dict(os.environ)
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    # shell=True: bunx/npx are shell shims on Windows.
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"ccusage failed ({proc.returncode}):\n{proc.stderr}")
    payload = json.loads(proc.stdout)

    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for day_entry in payload.get("daily", []):
        # ccusage calls the bucket key "period" (older versions: "date").
        day = day_entry.get("period") or day_entry["date"]
        for breakdown in day_entry.get("modelBreakdowns", []):
            model = breakdown["modelName"]
            bucket = totals[(day, model)]
            for _, cc_field in FIELDS:
                bucket[cc_field] += int(breakdown.get(cc_field, 0))
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timezone", default="UTC", help="IANA timezone for daily buckets")
    parser.add_argument(
        "--keep-json", action="store_true", help="Dump both aggregates next to this script"
    )
    args = parser.parse_args()
    tz = ZoneInfo(args.timezone)
    if args.timezone != "UTC":
        print(
            "WARNING: ccusage v20 ignores --timezone for daily bucketing; "
            "day-boundary diffs are expected outside UTC.",
            file=sys.stderr,
        )

    live_projects = Path.home() / ".claude" / "projects"
    if not live_projects.exists():
        raise SystemExit(f"{live_projects} not found.")

    with tempfile.TemporaryDirectory(prefix="reconcile-ccusage-") as tmp:
        frozen_config = Path(tmp) / "claude"
        frozen_projects = frozen_config / "projects"
        print(f"freezing {live_projects} -> {frozen_projects} ...")
        shutil.copytree(live_projects, frozen_projects)

        ours = collect_ours(frozen_projects, tz)
        theirs = collect_ccusage(frozen_config, args.timezone)

    if args.keep_json:
        dump = {
            "ours": {f"{d}|{m}": dict(c) for (d, m), c in ours.items()},
            "ccusage": {f"{d}|{m}": dict(c) for (d, m), c in theirs.items()},
        }
        out = Path(__file__).with_name("reconcile_dump.json")
        out.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    all_keys = sorted(set(ours) | set(theirs))
    matched = 0
    explained: list[str] = []
    unexplained: list[str] = []

    for day, model in all_keys:
        our_counts = ours.get((day, model), Counter())
        cc_counts = theirs.get((day, model), Counter())
        diffs: list[str] = []
        for our_fields, cc_field in FIELDS:
            our_value = sum(our_counts.get(f, 0) for f in our_fields)
            cc_value = cc_counts.get(cc_field, 0)
            if our_value != cc_value:
                diffs.append(
                    f"{cc_field}: ours={our_value} ccusage={cc_value} (d={our_value - cc_value:+})"
                )
        if not diffs:
            matched += 1
            continue
        line = f"{day}  {model:<30} " + "; ".join(diffs)
        if model == SYNTHETIC and (day, model) not in theirs:
            explained.append(line + "  [explained: ccusage skips <synthetic> messages]")
        elif model == "(unknown)" and (day, model) not in theirs:
            explained.append(line + "  [explained: records without a model field]")
        else:
            unexplained.append(line)

    print()
    print("=== Reconciliation: prompt-analytics vs ccusage daily ===")
    print(f"timezone:        {args.timezone}")
    print(f"(date, model) buckets compared: {len(all_keys)}")
    print(f"exact matches:   {matched}")
    print(f"explained:       {len(explained)}")
    for line in explained:
        print(f"  {line}")
    print(f"UNEXPLAINED:     {len(unexplained)}")
    for line in unexplained:
        print(f"  {line}")

    grand_ours: Counter[str] = Counter()
    grand_cc: Counter[str] = Counter()
    for counts in ours.values():
        grand_ours.update(counts)
    for counts in theirs.values():
        grand_cc.update(counts)
    print()
    print("Grand totals (all days, all models):")
    for our_fields, cc_field in FIELDS:
        our_value = sum(grand_ours.get(f, 0) for f in our_fields)
        print(f"  {cc_field:<22} ours={our_value:>14,}  ccusage={grand_cc.get(cc_field, 0):>14,}")

    if unexplained:
        print("\nFAIL: unexplained discrepancies above.")
        return 1
    print("\nOK: all buckets match exactly or are explained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
