"""CLI wrapper around the CSV tide importer (Phase 2 stub).

Usage (Phase 2)::

    python -m scripts.import_tide_csv --station PH-NAMRIA-001 path/to/tides.csv
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a NAMRIA-style tide CSV.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--station", required=True, help="Tide station code")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        f"Tide CSV import is implemented in Phase 2 "
        f"(station={args.station}, file={args.csv_path}).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
