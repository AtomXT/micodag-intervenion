#!/usr/bin/env python3
"""Run the practical local setup check without saving experiment results."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# This diagnostic promises no repository outputs. Prevent project imports and
# the official DCDI child from leaving bytecode caches in a fresh checkout.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_main_experiment as main_experiment
from src.main_experiment_data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MASTER_SEED,
    MAX_MASTER_SEED,
    MainExperimentDataError,
    load_main_experiment_instance,
    load_main_experiment_manifest,
)
from src.utils import interventional_cpdag


TEST_P = 10
TEST_EDGE_MULTIPLIER = 1
TEST_REPLICATE = 1
DCDI_METHODS = ("dcdi_g_unknown", "dcdi_g_oracle")
SETUP_METHOD_ORDER = tuple(
    method
    for method in main_experiment.METHODS
    if method not in DCDI_METHODS and method != "bacadi_unknown"
) + DCDI_METHODS


@contextmanager
def _fit_progress(method: str, time_limit: float):
    """Make the intentionally long DCDI check visibly alive without saving logs."""
    source = main_experiment.OFFICIAL_IMPLEMENTATIONS.get(method)
    label = "project PS-MIP" if source is None else source["name"]
    detail = ""
    if method in DCDI_METHODS or method == "bacadi_unknown":
        detail = (
            f"; this slow fit can run until the {time_limit:g}-second limit"
        )
    print(f"\n[{method}] running {label}{detail}...", flush=True)

    stopped = threading.Event()

    def heartbeat():
        started = time.monotonic()
        while not stopped.wait(60):
            elapsed = time.monotonic() - started
            print(
                f"[{method}] still running ({elapsed:.0f} seconds elapsed)...",
                flush=True,
            )

    worker = None
    if method in DCDI_METHODS or method == "bacadi_unknown":
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
    try:
        yield
    finally:
        stopped.set()
        if worker is not None:
            worker.join(timeout=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the local main-experiment setup on p=10, e=1, replicate=1. "
            "The one-hour-capped BaCaDI fit is opt-in; results are printed and "
            "never saved."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"persisted main-experiment data (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help=f"dataset-suite master seed (default: {DEFAULT_MASTER_SEED})",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=main_experiment.DEFAULT_TIME_LIMIT,
        help="per-method fit limit in seconds (default: 3600)",
    )
    parser.add_argument(
        "--metric-time-limit",
        type=float,
        default=main_experiment.DEFAULT_TIME_LIMIT,
        help="per-method equivalence-metric limit in seconds (default: 3600)",
    )
    parser.add_argument(
        "--dcdi-root",
        type=Path,
        default=PROJECT_ROOT / "external" / "dcdi",
        help="hash-verified vendored snapshot of the official DCDI repository",
    )
    parser.add_argument(
        "--bacadi-root",
        type=Path,
        default=PROJECT_ROOT / "external" / "bacadi",
        help="hash-verified vendored snapshot of the official BaCaDI package",
    )
    parser.add_argument(
        "--include-bacadi",
        action="store_true",
        help=(
            "also run the official-repository-default BaCaDI configuration; this is "
            "disabled by "
            "default because one fit can use the complete one-hour cap"
        ),
    )
    parser.add_argument(
        "--rscript",
        default="Rscript",
        help="Rscript executable providing pcalg and SID for DCDI reporting",
    )
    return parser


def _arguments(
    namespace: argparse.Namespace, manifest: dict, methods
) -> SimpleNamespace:
    """Build the runner contract while deliberately omitting any output path."""
    return SimpleNamespace(
        methods=list(methods),
        data_root=namespace.data_root.expanduser().resolve(),
        seed=namespace.seed,
        time_limit=namespace.time_limit,
        metric_time_limit=namespace.metric_time_limit,
        dcdi_root=namespace.dcdi_root.expanduser().resolve(),
        bacadi_root=namespace.bacadi_root.expanduser().resolve(),
        rscript=namespace.rscript,
        generator_numpy_version=manifest["provenance"]["numpy_version"],
        official_runtime={},
        transient_artifacts=True,
    )


def _primary_setting(method: str, data) -> dict:
    expected = main_experiment.PRIMARY_SETTING_IDS[method]
    matches = [
        setting
        for setting in main_experiment._method_settings(method, data)
        if setting["setting_id"] == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"primary setting {expected!r} is not uniquely defined for {method}"
        )
    return matches[0]


def _print_method_result(method: str, row: dict, result: dict) -> None:
    source = main_experiment.OFFICIAL_IMPLEMENTATIONS.get(method)
    reference = "" if source is None else source.get(
        "source_version", source.get("tested_version", "version recorded at runtime")
    )
    source_label = (
        "project PS-MIP implementation"
        if source is None
        else f"{source['name']} — {source['source_url']} ({reference})"
    )
    print(f"\n[{method}] {row['status']}")
    print("Source:", source_label)
    print(
        "Setting:",
        {
            "setting_id": row["setting_id"],
            "tuning_parameter": row["tuning_parameter"],
            "tuning_value": row["tuning_value"],
            "target_setting": row["target_setting"],
            "graph_penalty": row["graph_penalty"],
            "target_penalty": row["target_penalty"],
            "screen_alpha": row["screen_alpha"],
            "screen_edges": row["screen_edges"],
            "screen_parent_sets": row["screen_parent_sets"],
        },
    )
    print("Estimated I-CPDAG:\n", np.asarray(result["estimated_icpdag"], dtype=int))
    if result.get("estimated_targets") is not None:
        print("Estimated environment targets:\n", result["estimated_targets"])
    if result.get("estimated_target_union") is not None:
        print("Estimated target union:", result["estimated_target_union"])
    print(
        "Performance:",
        {
            "d_cpdag": row["d_cpdag"],
            "FDP": row["fdp"],
            "TDP": row["tdp"],
            "target_hamming": row["target_hamming"],
            "union_target_error": row["union_target_error"],
        },
    )
    print(
        "Fit details:",
        {
            "fit_seconds": row["fit_seconds"],
            "metric_seconds": row["metric_seconds"],
            "objective": row["objective_value"],
            "mip_gap": row["mip_gap"],
            "solver_status": row["solver_status"],
            "optimal": row["optimal"],
        },
    )


def run_check(namespace: argparse.Namespace) -> int:
    """Return 0 only when every selected fit and metric path succeeds."""
    data_root = namespace.data_root.expanduser().resolve()
    manifest = load_main_experiment_manifest(
        data_root, master_seed=namespace.seed
    )
    method_order = list(SETUP_METHOD_ORDER)
    if getattr(namespace, "include_bacadi", False):
        method_order.insert(-len(DCDI_METHODS), "bacadi_unknown")
    args = _arguments(namespace, manifest, method_order)

    print(
        "Checking required packages, APIs, and official implementation sources...",
        flush=True,
    )
    args.official_runtime = main_experiment._validate_competitor_runtime(args)
    data, true_dag, true_targets, info = load_main_experiment_instance(
        data_root,
        TEST_P,
        TEST_EDGE_MULTIPLIER,
        TEST_REPLICATE,
        manifest=manifest,
        master_seed=namespace.seed,
    )
    truth = interventional_cpdag(true_dag, true_targets)

    print(
        "\nTest instance:",
        {
            "p": TEST_P,
            "edge_multiplier": TEST_EDGE_MULTIPLIER,
            "replicate": TEST_REPLICATE,
            "seed": info["seed"],
            "sample_sizes": [len(values) for values in data],
            "true_edges": int(true_dag.sum()),
            "data_sha256": info["data_sha256"],
        },
    )
    print("True I-CPDAG:\n", truth)
    print("True environment targets:\n", true_targets)

    failures = []
    for method in method_order:
        setting = _primary_setting(method, data)
        row = main_experiment._base_row(
            TEST_P,
            TEST_EDGE_MULTIPLIER,
            TEST_REPLICATE,
            info["seed"],
            method,
            setting,
            data,
            true_dag,
            true_targets,
            args,
            info["data_sha256"],
            info["instance_sha256"],
        )
        started = time.perf_counter()
        try:
            with _fit_progress(method, args.time_limit):
                result = main_experiment._run_method(
                    method,
                    data,
                    true_targets,
                    args,
                    setting,
                    TEST_P,
                    TEST_EDGE_MULTIPLIER,
                    TEST_REPLICATE,
                    info["seed"],
                )
            if result.get("artifact_dir") not in (None, ""):
                raise RuntimeError(
                    "the no-save setup check unexpectedly retained an artifact directory"
                )
            main_experiment._add_fit_result(row, result, true_targets)
        except Exception as error:
            row.update(
                status="fit_error",
                fit_seconds=time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )
            failures.append((method, row["error"]))
            print(f"\n[{method}] fit_error\nError: {row['error']}")
            continue

        metric_started = time.perf_counter()
        try:
            main_experiment._add_metrics(
                row,
                result["estimated_icpdag"],
                truth,
                args.metric_time_limit,
            )
        except Exception as error:
            row.update(
                status="metric_error",
                metric_seconds=time.perf_counter() - metric_started,
                error=f"{type(error).__name__}: {error}",
            )
            failures.append((method, row["error"]))
            print(f"\n[{method}] metric_error\nError: {row['error']}")
            continue

        row["status"] = (
            "ok_nonoptimal" if row["optimal"] is False else "ok"
        )
        _print_method_result(method, row, result)

    print("\nSummary:")
    print(f"  methods passed: {len(method_order) - len(failures)}")
    print(f"  methods failed: {len(failures)}")
    print("  saved outputs: none")
    for method, error in failures:
        print(f"  - {method}: {error}")
    return 1 if failures else 0


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if (
        not np.isfinite([args.time_limit, args.metric_time_limit]).all()
        or args.time_limit <= 0
        or args.metric_time_limit <= 0
    ):
        parser.error("time limits must be positive and finite")
    if not 0 <= args.seed <= MAX_MASTER_SEED:
        parser.error(f"seed must be between 0 and {MAX_MASTER_SEED}")

    # Keep incidental Matplotlib state out of the repository as well. DCDI
    # stages its method files in an automatically cleaned temporary directory;
    # the requested Python GIES package writes no method files.
    with tempfile.TemporaryDirectory(prefix="main-experiment-setup-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            return run_check(args)
        except MainExperimentDataError as error:
            print(f"main experiment data error: {error}", file=sys.stderr)
            return 2
        except main_experiment.OfficialRuntimeError as error:
            print(f"official baseline setup error: {error}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
