#!/usr/bin/env python3
"""Plot every main-experiment method that currently has result rows."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.aggregate_main_experiment import DEFAULT_OUTPUT, main as aggregate


def main() -> int:
    aggregate()
    print(f"\nGraph plot ready: {DEFAULT_OUTPUT / 'main_tdp_fdp.png'}")
    print(f"Target plot ready: {DEFAULT_OUTPUT / 'target_tpr_fpr.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
