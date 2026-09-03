#!/usr/bin/env python3
"""Prepare the exact pinned Sachs data package used by the UT-IGSP analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sachs_data import (
    DEFAULT_DATA_ROOT,
    SACHS_SAMPLE_COUNTS,
    SACHS_TOTAL_SAMPLES,
    SOURCE_COMMIT,
    SachsDataError,
    prepare_sachs_data,
    validate_sachs_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and validate the exact six Sachs files from the pinned "
            "UT-IGSP author repository, then create a pickle-free NPZ package."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="prepared dataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "offline directory containing iv=.txt, iv=1.txt, iv=3.txt, "
            "iv=4.txt, iv=6.txt, and iv=8.txt; omit to download over HTTPS"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing directory only when its manifest identifies this pinned Sachs package",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate the existing raw files, manifest, and archive without changing them",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="per-request HTTPS timeout in seconds (default: %(default)s)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.verify_only and args.overwrite:
        parser.error("--verify-only and --overwrite cannot be combined")
    if args.verify_only and args.source_dir is not None:
        parser.error("--verify-only and --source-dir cannot be combined")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    data_root = args.output_dir.expanduser().resolve()
    try:
        if args.verify_only:
            info = validate_sachs_data(data_root)
            action = "Validated"
        else:
            info = prepare_sachs_data(
                data_root,
                source_directory=args.source_dir,
                overwrite=args.overwrite,
                timeout=args.timeout,
            )
            action = "Ready"
    except SachsDataError as error:
        print("Sachs data error: %s" % error, file=sys.stderr)
        return 2

    print(
        "%s: %d samples in environments %s under %s"
        % (action, SACHS_TOTAL_SAMPLES, list(SACHS_SAMPLE_COUNTS), data_root)
    )
    print("Pinned UT-IGSP source commit: %s" % SOURCE_COMMIT)
    print("Data SHA-256: %s" % info["data_sha256"])
    print("Archive SHA-256: %s" % info["archive_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

