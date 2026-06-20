"""Command-line interface for prompt-analytics-for-claude-code.

Two families of subcommands:

* **Analysis** (work on the fly, no prior ``extract`` needed -- the output
  CSVs are only used as a cache when fresh): ``summary``, ``by-project``,
  ``by-model``, ``by-token-type``, ``by-category``, ``prompts``, ``sessions``,
  ``compare``, ``export``.
* **Pipeline**: ``extract`` (write the CSVs), ``run`` (extract + snapshot +
  optional categorize), ``snapshot``, ``categorize``, ``config``,
  ``dashboard``.

Heavy imports stay inside the handlers so ``--help`` is instant.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .analytics import Dataset

__all__ = ["build_parser", "main"]

DEFAULT_OUTPUT_DIR = "./output"
OUTPUT_DIR_ENV = "PROMPT_ANALYTICS_OUTPUT_DIR"

_EXAMPLES = """\
examples:
  prompt-analytics summary                       totals, tokens and cost in the terminal
  prompt-analytics by-project                    where the money goes (cumulative %)
  prompt-analytics timeline --by month           cost per month
  prompt-analytics summary --since 2026-06-01    analyze one period (any analysis command)
  prompt-analytics by-token-type                 context rent vs generation in the bill
  prompt-analytics by-output                     what Claude produced: language mix, code vs tests
  prompt-analytics by-context                    what fills the cache: loading vs rent per source
  prompt-analytics sessions --depth              marginal prompt cost vs session depth
  prompt-analytics context                       accumulated context per turn (when to /compact)
  prompt-analytics ttl                           what pauses cost in cache re-writes
  prompt-analytics compactions                   context before/after /compact, rebuild cost
  prompt-analytics overhead                      fixed per-session cost (system+CLAUDE.md+MCP)
  prompt-analytics model-category --whatif claude-sonnet-4-6
  prompt-analytics recommend                     what compacting long sessions earlier would save
  prompt-analytics burn-rate                     $/day and week-over-week trend
  prompt-analytics break-even                    your API-equivalent value vs a Max subscription
  prompt-analytics prompts --top 10              your 10 most expensive prompts
  prompt-analytics compare --providers anthropic,copilot
  prompt-analytics summary --format json         machine-readable output (also: csv)
  prompt-analytics summary --from-csv demo_data  analyze a CSV export as-is
  prompt-analytics extract --output-dir ./out    write the normalized CSVs
  prompt-analytics export --flat                 one denormalized CSV for Excel/BI
"""


def _add_output_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are written/read (default: %(default)s).",
    )


def _analytics_parent() -> argparse.ArgumentParser:
    """Shared flags of every analysis command (7.3/7.4)."""
    parent = argparse.ArgumentParser(add_help=False)
    _add_output_dir(parent)
    parent.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Output format (default: %(default)s).",
    )
    parent.add_argument(
        "--pricing",
        metavar="PATH",
        help="Custom pricing YAML (default: the bundled pricing.yml).",
    )
    parent.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached CSVs and the parse cache; re-parse the JSONL history.",
    )
    parent.add_argument(
        "--from-csv",
        metavar="DIR",
        dest="from_csv",
        help="Analyze the CSVs in DIR as-is (no live parse, no freshness check "
        "against ~/.claude/projects) -- e.g. a demo dataset or an archived extract.",
    )
    parent.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Only analyze prompts dated on or after this day (inclusive).",
    )
    parent.add_argument(
        "--until",
        metavar="YYYY-MM-DD",
        help="Only analyze prompts dated on or before this day (inclusive).",
    )
    return parent


def _provider_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        metavar="NAME",
        default="anthropic",
        help="Pricing provider for the cost column, e.g. anthropic or copilot "
        "(default: %(default)s).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="prompt-analytics",
        description="Prompt-level analytics for Claude Code, from your local JSONL files.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    analytics_parent = _analytics_parent()

    # --- Analysis commands (7.3) -------------------------------------------
    p_summary = subparsers.add_parser(
        "summary",
        parents=[analytics_parent],
        help="Overview: sessions, prompts, tokens by type, cost per provider.",
    )
    p_summary.set_defaults(func=_handle_summary)

    p_by_project = subparsers.add_parser(
        "by-project",
        parents=[analytics_parent],
        help="Cost/tokens/prompts per project, sorted by cost.",
    )
    _provider_arg(p_by_project)
    # Deprecated: the cumulative %% column is now always shown. Kept as an
    # accepted no-op so published `by-project --pareto` invocations don't break.
    p_by_project.add_argument(
        "--pareto",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p_by_project.set_defaults(func=_handle_by_project)

    p_by_model = subparsers.add_parser(
        "by-model",
        parents=[analytics_parent],
        help="Token and cost split per model.",
    )
    _provider_arg(p_by_model)
    p_by_model.add_argument(
        "--compact",
        action="store_true",
        help="Narrow view (abbreviated token counts, no input/output) for 80-column terminals.",
    )
    p_by_model.set_defaults(func=_handle_by_model)

    p_by_token_type = subparsers.add_parser(
        "by-token-type",
        parents=[analytics_parent],
        help="Token volume and cost split per token type (context rent vs generation).",
    )
    _provider_arg(p_by_token_type)
    p_by_token_type.set_defaults(func=_handle_by_token_type)

    p_by_category = subparsers.add_parser(
        "by-category",
        parents=[analytics_parent],
        help="Cost/prompt split per LLM-assigned category (needs `categorize`).",
    )
    _provider_arg(p_by_category)
    p_by_category.set_defaults(func=_handle_by_category)

    p_by_output = subparsers.add_parser(
        "by-output",
        parents=[analytics_parent],
        help="Output composition: language mix, code vs tests, prose vs code cost, lines produced.",
    )
    _provider_arg(p_by_output)
    p_by_output.set_defaults(func=_handle_by_output)

    p_by_context = subparsers.add_parser(
        "by-context",
        parents=[analytics_parent],
        help="Context cost over time: what fills the cache, loading vs rent per source.",
    )
    _provider_arg(p_by_context)
    p_by_context.set_defaults(func=_handle_by_context)

    p_prompts = subparsers.add_parser(
        "prompts",
        parents=[analytics_parent],
        help="The N most expensive prompts, with preview.",
    )
    _provider_arg(p_prompts)
    p_prompts.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="How many prompts to show (default: %(default)s).",
    )
    p_prompts.set_defaults(func=_handle_prompts)

    p_sessions = subparsers.add_parser(
        "sessions",
        parents=[analytics_parent],
        help="Sessions ranked by cost, or --depth for the depth meta-analysis.",
    )
    _provider_arg(p_sessions)
    sessions_mode = p_sessions.add_mutually_exclusive_group()
    sessions_mode.add_argument(
        "--depth",
        action="store_true",
        help="Marginal prompt cost and cache mix by position in the session.",
    )
    sessions_mode.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="How many sessions to list (default: %(default)s; 0 = all).",
    )
    p_sessions.add_argument(
        "--project",
        metavar="NAME",
        help="Restrict to one project (matches sessions.csv project; --depth too).",
    )
    p_sessions.set_defaults(func=_handle_sessions)

    # --- Request-grain analyses (phase 2: TTL, compaction, context) ---------
    p_context = subparsers.add_parser(
        "context",
        parents=[analytics_parent],
        help="Accumulated context per turn by session depth (the 'time to /compact' signal).",
    )
    _provider_arg(p_context)
    p_context.set_defaults(func=_handle_context)

    p_ttl = subparsers.add_parser(
        "ttl",
        parents=[analytics_parent],
        help="Cache TTL expiry losses: what pauses cost in cache re-writes.",
    )
    _provider_arg(p_ttl)
    p_ttl.set_defaults(func=_handle_ttl)

    p_compactions = subparsers.add_parser(
        "compactions",
        parents=[analytics_parent],
        help="Compaction events: context before/after and cache rebuild cost.",
    )
    _provider_arg(p_compactions)
    p_compactions.set_defaults(func=_handle_compactions)

    p_overhead = subparsers.add_parser(
        "overhead",
        parents=[analytics_parent],
        help="Fixed per-session overhead (system prompt + CLAUDE.md + MCP tools).",
    )
    _provider_arg(p_overhead)
    p_overhead.set_defaults(func=_handle_overhead)

    # --- Cross-model, recommendations, burn rate (2.5-2.7) ------------------
    p_model_cat = subparsers.add_parser(
        "model-category",
        parents=[analytics_parent],
        help="Cost crossed by model x category, with an optional re-pricing what-if.",
    )
    _provider_arg(p_model_cat)
    p_model_cat.add_argument(
        "--whatif",
        metavar="MODEL",
        help="Re-price every cell on MODEL at identical token counts "
        "(e.g. --whatif claude-sonnet-4-6).",
    )
    p_model_cat.set_defaults(func=_handle_model_category)

    p_recommend = subparsers.add_parser(
        "recommend",
        parents=[analytics_parent],
        help="Prescriptive estimate: what compacting long sessions earlier would save.",
    )
    _provider_arg(p_recommend)
    p_recommend.add_argument(
        "--min-prompts",
        type=int,
        default=50,
        metavar="N",
        help="Only consider sessions longer than N prompts (default: %(default)s).",
    )
    p_recommend.add_argument(
        "--compact-at",
        type=int,
        default=30,
        metavar="K",
        help="Hypothetical compaction depth (default: %(default)s).",
    )
    p_recommend.set_defaults(func=_handle_recommend)

    p_burn = subparsers.add_parser(
        "burn-rate",
        parents=[analytics_parent],
        help="Spend trend: $/day and week-over-week.",
    )
    _provider_arg(p_burn)
    p_burn.add_argument(
        "--weeks",
        type=int,
        default=8,
        metavar="N",
        help="How many recent weeks to list (default: %(default)s; 0 = all).",
    )
    p_burn.set_defaults(func=_handle_burn_rate)

    p_timeline = subparsers.add_parser(
        "timeline",
        parents=[analytics_parent],
        help="Cost, prompts and tokens grouped by day, week or month.",
    )
    _provider_arg(p_timeline)
    p_timeline.add_argument(
        "--by",
        choices=("day", "week", "month"),
        default="day",
        help="Grouping granularity (default: %(default)s).",
    )
    p_timeline.set_defaults(func=_handle_timeline)

    p_break_even = subparsers.add_parser(
        "break-even",
        parents=[analytics_parent],
        help="Plan break-even: your API-equivalent value vs a flat-rate subscription.",
    )
    _provider_arg(p_break_even)
    p_break_even.set_defaults(func=_handle_break_even)

    p_compare = subparsers.add_parser(
        "compare",
        parents=[analytics_parent],
        help="The same usage priced on several provider grids.",
    )
    p_compare.add_argument(
        "--providers",
        default="anthropic,copilot",
        metavar="A,B",
        help="Comma-separated pricing providers (default: %(default)s).",
    )
    p_compare.set_defaults(func=_handle_compare)

    p_export = subparsers.add_parser(
        "export",
        parents=[analytics_parent],
        help="Denormalized exports for Excel/BI.",
    )
    export_mode = p_export.add_mutually_exclusive_group(required=True)
    export_mode.add_argument(
        "--flat",
        action="store_true",
        help="One row per prompt: pivoted token columns, per-provider costs, "
        "duplicated session info.",
    )
    p_export.add_argument(
        "--out",
        metavar="PATH",
        help="Output file (default: <output-dir>/flat.csv).",
    )
    p_export.set_defaults(func=_handle_export)

    # --- Pipeline commands ---------------------------------------------------
    p_extract = subparsers.add_parser(
        "extract", help="Extract usage data from local JSONL files into CSVs."
    )
    p_extract.add_argument(
        "--no-text",
        action="store_true",
        help="Do not store prompt/response text.",
    )
    _add_output_dir(p_extract)
    p_extract.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Only include records on or after this date.",
    )
    p_extract.add_argument(
        "--until",
        metavar="YYYY-MM-DD",
        help="Only include records on or before this date (inclusive).",
    )
    p_extract.add_argument(
        "--timezone",
        metavar="IANA_NAME",
        help="Timezone for date bounds and session dates (default: local).",
    )
    p_extract.add_argument(
        "--strict",
        action="store_true",
        help="Treat extraction warnings as errors (non-zero exit code).",
    )
    p_extract.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the per-file parse cache and re-parse everything.",
    )
    p_extract.set_defaults(func=_handle_extract)

    # snapshot
    p_snapshot = subparsers.add_parser(
        "snapshot", help="Append the current quota utilization to quota_log.csv."
    )
    _add_output_dir(p_snapshot)
    p_snapshot.set_defaults(func=_handle_snapshot)

    # categorize
    p_categorize = subparsers.add_parser(
        "categorize",
        help="Categorize prompts (heuristic by default, --llm for LLM).",
    )
    _add_output_dir(p_categorize)
    p_categorize.add_argument(
        "--llm",
        action="store_true",
        help="Use an LLM API instead of the heuristic classifier.",
    )
    p_categorize.add_argument(
        "--batch",
        action="store_true",
        help="Use the Anthropic Message Batches API (--llm only, -50%% cost).",
    )
    p_categorize.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "anthropic", "openrouter", "ollama"],
        help="LLM provider (--llm only, default: auto).",
    )
    p_categorize.add_argument(
        "--model",
        default="",
        help="Override the default model for the chosen provider.",
    )
    p_categorize.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Checkpoint every N classifications (default: %(default)s).",
    )
    p_categorize.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds between LLM calls (default: %(default)s).",
    )
    p_categorize.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N prompts (0 = all).",
    )
    p_categorize.set_defaults(func=_handle_categorize)

    # run
    p_run = subparsers.add_parser(
        "run", help="Run the full pipeline (extract + optional categorize + snapshot)."
    )
    p_run.add_argument(
        "--categorize",
        action="store_true",
        help="Run the categorization step as part of the pipeline.",
    )
    p_run.add_argument(
        "--llm",
        action="store_true",
        help="Categorize: use an LLM API instead of the heuristic classifier.",
    )
    p_run.add_argument(
        "--batch",
        action="store_true",
        help="Categorize: use the Anthropic Message Batches API (--llm only, -50%% cost).",
    )
    p_run.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "anthropic", "openrouter", "ollama"],
        help="Categorize: LLM provider (--llm only, default: auto).",
    )
    p_run.add_argument(
        "--model",
        default="",
        help="Categorize: override the default model for the chosen provider.",
    )
    p_run.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Categorize: checkpoint every N prompts (default: %(default)s).",
    )
    p_run.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Categorize: seconds between LLM calls (default: %(default)s).",
    )
    p_run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Categorize: process at most N prompts (0 = all).",
    )
    p_run.add_argument(
        "--no-text",
        action="store_true",
        help="Do not store prompt/response text.",
    )
    _add_output_dir(p_run)
    p_run.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Only include records on or after this date.",
    )
    p_run.add_argument(
        "--until",
        metavar="YYYY-MM-DD",
        help="Only include records on or before this date (inclusive).",
    )
    p_run.add_argument(
        "--timezone",
        metavar="IANA_NAME",
        help="Timezone for date bounds and session dates (default: local).",
    )
    p_run.add_argument(
        "--strict",
        action="store_true",
        help="Treat extraction warnings as errors (non-zero exit code).",
    )
    p_run.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the per-file parse cache and re-parse everything.",
    )
    p_run.set_defaults(func=_handle_run)

    # config
    p_config = subparsers.add_parser("config", help="Manage the config.yml configuration file.")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    p_config_init = config_sub.add_parser(
        "init", help="Write the default config.yml into the output directory."
    )
    _add_output_dir(p_config_init)
    p_config_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.yml.",
    )
    p_config_init.set_defaults(func=_handle_config_init)

    # dashboard
    p_dashboard = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard.")
    _add_output_dir(p_dashboard)
    p_dashboard.set_defaults(func=_handle_dashboard)

    return parser


# ---------------------------------------------------------------------------
# Analysis handlers (7.3): load the dataset on the fly, render, exit code.
# ---------------------------------------------------------------------------


def _pricing_path(args: argparse.Namespace) -> Path | None:
    from pathlib import Path

    return Path(args.pricing) if getattr(args, "pricing", None) else None


def _load_dataset(args: argparse.Namespace) -> Dataset:
    from pathlib import Path

    from . import analytics

    # --from-csv: trust the given CSVs as-is, like the dashboard does (D1).
    if getattr(args, "from_csv", None):
        return analytics.dataset_from_csvs(Path(args.from_csv), pricing_path=_pricing_path(args))
    return analytics.load_dataset(
        Path(args.output_dir),
        use_cache=not args.no_cache,
        pricing_path=_pricing_path(args),
    )


def _apply_date_window(args: argparse.Namespace, ds: Dataset) -> tuple[Dataset | None, int]:
    """Narrow ``ds`` to --since/--until; validate the bounds (exit 2 on a bad date)."""
    import sys

    from . import analytics
    from .extract import _parse_bound

    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    if not since and not until:
        return ds, 0
    try:
        if since:
            _parse_bound(since, "since")
        if until:
            _parse_bound(until, "until")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None, 2
    ds = analytics.filter_dates(ds, since, until)
    if not ds.tokens and not ds.prompts:
        window = " .. ".join(b or "..." for b in (since, until))
        print(f"No data in the date range ({window}).", file=sys.stderr)
        return None, 1
    return ds, 0


def _dataset_or_fail(args: argparse.Namespace) -> tuple[Dataset | None, int]:
    """Load the dataset; on empty data print a hint and return exit code 1."""
    import sys

    from .paths import claude_projects_dir
    from .pricing import PricingError

    try:
        ds = _load_dataset(args)
    except PricingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None, 2
    if not ds.tokens and not ds.prompts:
        if getattr(args, "from_csv", None):
            print(f"No data found in the CSVs at {args.from_csv}.", file=sys.stderr)
        else:
            claude_dir = claude_projects_dir()
            print(
                f"No Claude Code data found (looked in {claude_dir} and {args.output_dir}).",
                file=sys.stderr,
            )
        return None, 1
    return _apply_date_window(args, ds)


def _check_providers(providers: list[str], pricing_path: Path | None) -> int:
    """0 when every provider exists in the pricing file, else 2 (with message)."""
    import sys

    from . import analytics
    from .pricing import PricingError

    try:
        known = analytics.known_providers(pricing_path)
    except PricingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    unknown = [p for p in providers if p not in known]
    if unknown:
        print(
            f"Error: unknown pricing provider(s): {', '.join(unknown)} "
            f"(known: {', '.join(known)}).",
            file=sys.stderr,
        )
        return 2
    return 0


def _handle_summary(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([], _pricing_path(args)):
        return code  # validates the pricing file itself
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.summary(ds), args.format)
    return 0


def _handle_by_project(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.by_project(ds, args.provider), args.format)
    return 0


def _handle_by_model(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.by_model(ds, args.provider, compact=args.compact), args.format)
    return 0


def _handle_by_token_type(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.by_token_type(ds, args.provider), args.format)
    return 0


def _handle_by_category(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.by_category(ds, args.provider), args.format)
    return 0


def _handle_by_output(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.by_output(ds, args.provider), args.format)
    return 0


def _handle_by_context(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.by_context(ds, args.provider), args.format)
    return 0


def _handle_prompts(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.top_prompts(ds, args.provider, top=args.top), args.format)
    return 0


def _handle_sessions(args: argparse.Namespace) -> int:
    import sys

    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    if args.project:
        ds = analytics.filter_project(ds, args.project)
        if not ds.sessions:
            print(f"No sessions found for project {args.project!r}.", file=sys.stderr)
            return 1
    if args.depth:
        result = analytics.session_depth(ds, args.provider)
    else:
        result = analytics.sessions_table(ds, args.provider, top=args.top)
    render(result, args.format)
    return 0


def _handle_context(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.context_growth(ds, args.provider), args.format)
    return 0


def _handle_ttl(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.ttl_losses(ds, args.provider), args.format)
    return 0


def _handle_compactions(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.compactions(ds, args.provider), args.format)
    return 0


def _handle_overhead(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.session_overhead(ds, args.provider), args.format)
    return 0


def _handle_model_category(args: argparse.Namespace) -> int:
    import sys

    from . import analytics
    from .pricing import get_model_pricing
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    # A what-if on an unpriced target would silently read as $0 (huge fake
    # saving) -- refuse it loudly instead.
    if args.whatif and get_model_pricing(args.whatif, args.provider, _pricing_path(args)) is None:
        print(
            f"Error: no {args.provider} pricing for --whatif model {args.whatif!r}.",
            file=sys.stderr,
        )
        return 2
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.model_category(ds, args.provider, target_model=args.whatif), args.format)
    return 0


def _handle_recommend(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(
        analytics.recommendations(
            ds, args.provider, min_prompts=args.min_prompts, compact_at=args.compact_at
        ),
        args.format,
    )
    return 0


def _handle_burn_rate(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.burn_rate(ds, args.provider, weeks=args.weeks), args.format)
    return 0


def _handle_timeline(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.timeline(ds, args.provider, by=args.by), args.format)
    return 0


def _read_quota_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    """Read quota_log.csv from the data dir (3.1 enrichment), or []."""
    import csv
    from pathlib import Path

    base = Path(args.from_csv) if getattr(args, "from_csv", None) else Path(args.output_dir)
    path = base / "quota_log.csv"
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def _handle_break_even(args: argparse.Namespace) -> int:
    from . import analytics
    from .render import render

    if code := _check_providers([args.provider], _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(
        analytics.break_even(ds, provider=args.provider, quota_rows=_read_quota_rows(args)),
        args.format,
    )
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    import sys

    from . import analytics
    from .render import render

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    if len(providers) < 2:
        print(
            "Error: --providers needs at least two comma-separated providers.",
            file=sys.stderr,
        )
        return 2
    if code := _check_providers(providers, _pricing_path(args)):
        return code
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    render(analytics.compare_providers(ds, providers), args.format)
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import analytics
    from .storage import atomic_write_csv

    if code := _check_providers([], _pricing_path(args)):
        return code  # validates the pricing file itself
    ds, code = _dataset_or_fail(args)
    if ds is None:
        return code
    columns, rows = analytics.flat_export(ds)
    out_path = Path(args.out) if args.out else Path(args.output_dir) / "flat.csv"
    atomic_write_csv(out_path, columns, rows)
    print(f"Wrote {len(rows)} rows ({len(columns)} columns) to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Pipeline handlers.
# ---------------------------------------------------------------------------

_NEXT_STEPS = """\
Next steps:
  prompt-analytics summary              totals in the terminal
  prompt-analytics by-project           where the money goes
  prompt-analytics timeline --by month  cost per month
  prompt-analytics sessions --depth     what a prompt costs deep into a session
  prompt-analytics dashboard            the full picture (install the [dashboard] extra)"""


def _print_epilogue(output_dir: Path) -> None:
    """Mini summary + next steps after extract/run (7.6); never fatal."""
    import sys

    from . import analytics
    from .pricing import PricingError

    try:
        ds = analytics.load_dataset(output_dir)
        for line in analytics.mini_summary(ds):
            print(line)
    except (PricingError, OSError) as exc:  # the extraction itself succeeded
        print(f"(mini summary unavailable: {exc})", file=sys.stderr)
    print(_NEXT_STEPS)


def _handle_extract(args: argparse.Namespace) -> int:
    """Dispatch the ``extract`` subcommand."""
    import sys
    from pathlib import Path

    from . import extract

    try:
        report = extract.run_extract(
            Path(args.output_dir),
            no_text=args.no_text,
            since=args.since,
            until=args.until,
            timezone_name=args.timezone,
            use_cache=not args.no_cache,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except PermissionError as exc:
        path_hint = getattr(exc, "filename", None) or "a CSV file"
        print(
            f"Error: cannot write {path_hint} — close any app (e.g. Excel)"
            " that has it open, then retry.",
            file=sys.stderr,
        )
        return 2
    for line in report.format_lines():
        print(line)
    if report.prompts:
        _print_epilogue(Path(args.output_dir))
    return report.exit_code(strict=args.strict)


def _handle_snapshot(args: argparse.Namespace) -> int:
    """Dispatch the ``snapshot`` subcommand."""
    from pathlib import Path

    from . import snapshot

    snapshot.run_snapshot(Path(args.output_dir))
    return 0


def _handle_categorize(args: argparse.Namespace) -> int:
    """Dispatch the ``categorize`` subcommand."""
    from . import categorize

    count = categorize.run_categorize(
        output_dir=args.output_dir,
        use_llm=args.llm,
        use_batch=args.batch,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        delay=args.delay,
        limit=args.limit,
    )
    return 0 if count >= 0 else 1


def _handle_run(args: argparse.Namespace) -> int:
    """Dispatch the ``run`` subcommand (full pipeline)."""
    import sys
    from pathlib import Path

    from . import extract, snapshot

    output_dir = Path(args.output_dir)

    # --- Step 1/3: extract ---
    print("=== Step 1/3: extract ===")
    try:
        report = extract.run_extract(
            output_dir,
            no_text=args.no_text,
            since=args.since,
            until=args.until,
            timezone_name=args.timezone,
            use_cache=not args.no_cache,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except PermissionError as exc:
        path_hint = getattr(exc, "filename", None) or "a CSV file"
        print(
            f"Error: cannot write {path_hint} — close any app (e.g. Excel)"
            " that has it open, then retry.",
            file=sys.stderr,
        )
        return 2
    for line in report.format_lines():
        print(line)
    sessions = report.sessions

    # --- Step 2/3: snapshot ---
    print("=== Step 2/3: snapshot ===")
    snapshot_rows = snapshot.run_snapshot(output_dir)
    print(f"Quota fields snapshotted: {snapshot_rows}")

    # --- Step 3/3: categorize ---
    print("=== Step 3/3: categorize ===")
    categorized = 0
    if not args.categorize:
        print("Categorization skipped (not requested).")
    else:
        from . import categorize

        # -1 means "could not attempt" (no prompts.csv); the extract step just
        # ran, so treat it as 0 for the pipeline summary. --llm/--provider/--batch
        # are forwarded so `run --categorize` reaches the same pipeline as the
        # standalone `categorize` command (3.4).
        categorized = max(
            0,
            categorize.run_categorize(
                output_dir=str(output_dir),
                use_llm=getattr(args, "llm", False),
                use_batch=getattr(args, "batch", False),
                provider=getattr(args, "provider", "auto"),
                model=getattr(args, "model", ""),
                batch_size=getattr(args, "batch_size", 50),
                delay=getattr(args, "delay", 0.1),
                limit=getattr(args, "limit", 0),
            ),
        )

    print(
        f"Summary: {sessions} sessions extracted, "
        f"{snapshot_rows} quota fields snapshotted, "
        f"{categorized} prompts categorized."
    )
    if report.prompts:
        _print_epilogue(output_dir)
    return report.exit_code(strict=args.strict)


def _handle_config_init(args: argparse.Namespace) -> int:
    """Dispatch the ``config init`` subcommand."""
    from pathlib import Path

    from .config import write_default_config

    output_dir = Path(args.output_dir)
    config_path = output_dir / "config.yml"
    existed = config_path.exists()
    written = write_default_config(output_dir, overwrite=args.force)
    if existed and not args.force:
        print(f"config.yml already exists at {written} (use --force to overwrite).")
    else:
        print(f"Wrote default configuration to {written}")
    return 0


def _handle_dashboard(args: argparse.Namespace) -> int:
    """Dispatch the ``dashboard`` subcommand.

    Resolves the output directory, exposes it to the dashboard process via the
    ``PROMPT_ANALYTICS_OUTPUT_DIR`` environment variable, and launches the
    Streamlit app located in the installed package.
    """
    import importlib.resources
    import os
    import subprocess
    import sys
    from pathlib import Path

    output_dir = Path(args.output_dir).resolve()

    # Locate app.py via importlib.resources so it works from an installed wheel.
    pkg_files = importlib.resources.files("prompt_analytics.dashboard")
    app_path = Path(str(pkg_files)) / "app.py"

    env = dict(os.environ)
    env[OUTPUT_DIR_ENV] = str(output_dir)

    # Forward any extra flags (e.g. --server.port 8599) straight to Streamlit;
    # `streamlit run app.py` accepts config options after the script path.
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    cmd += getattr(args, "streamlit_args", None) or []

    result = subprocess.run(cmd, env=env, check=False)
    return int(result.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the matching subcommand handler.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    import contextlib
    import sys

    # On Windows < 3.15 (PEP 686), a *redirected or piped* stdout/stderr is
    # encoded in the locale codepage (cp1252) -- prompt previews are arbitrary
    # UTF-8, so `prompts --format json > out.json` would crash with a
    # UnicodeEncodeError on the first emoji. Force UTF-8 on both streams;
    # AttributeError covers non-reconfigurable shims (e.g. a plain StringIO).
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
            with contextlib.suppress(AttributeError, OSError):
                stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = build_parser()
    # ``dashboard`` forwards unknown flags (e.g. --server.port) to Streamlit, so
    # parse leniently; every other command stays strict -- re-parse to emit
    # argparse's standard "unrecognized arguments" error (exit 2) for them.
    args, extra = parser.parse_known_args(argv)
    if extra and getattr(args, "func", None) is not _handle_dashboard:
        parser.parse_args(argv)
    args.streamlit_args = extra
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
