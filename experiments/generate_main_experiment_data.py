#!/usr/bin/env python3
"""Generate every persisted dataset required by the main experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main_experiment_data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MASTER_SEED,
    MAX_MASTER_SEED,
    MainExperimentDataError,
    expected_instance_keys,
    generate_main_experiment_data,
    validate_main_experiment_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the complete persisted dataset suite for the main "
            "synthetic experiment."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help=f"master seed (default: {DEFAULT_MASTER_SEED})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing mismatched dataset suite",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate all archives without generating or changing data",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.seed <= MAX_MASTER_SEED:
        parser.error(f"--seed must be between 0 and {MAX_MASTER_SEED}")
    if args.verify_only and args.overwrite:
        parser.error("--verify-only and --overwrite cannot be combined")
    data_root = args.output_dir.expanduser().resolve()
    try:
        if args.verify_only:
            validate_main_experiment_data(data_root, master_seed=args.seed)
            action = "Validated"
        else:
            generate_main_experiment_data(
                data_root,
                master_seed=args.seed,
                overwrite=args.overwrite,
            )
            action = "Ready"
    except MainExperimentDataError as error:
        print(f"main experiment data error: {error}", file=sys.stderr)
        return 2
    print(
        f"{action}: {len(expected_instance_keys())} main-experiment instances "
        f"under {data_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
