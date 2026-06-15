"""Module entry point for ``python -m prompt_analytics``."""

from __future__ import annotations

import sys

__all__ = ["main"]


def main() -> None:
    """Run the command-line interface."""
    from .cli import main as cli_main

    sys.exit(cli_main())


if __name__ == "__main__":
    main()
