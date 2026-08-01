#!/usr/bin/env python3
"""Run checkpointed PS-MIP, GnIES, and UT-IGSP experiments."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "SyntheticData"
ALPHA = 0.1
CONFIGURATION_LIMIT = 1_000_000
MIP_GRIDS = {
    1: [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.32],
    2: [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.32, 0.64],
    3: [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.32, 0.64],
}
GNIES_GRIDS = {
    1: [0.005, 0.01, 0.03, 0.1, 0.3, 1, 3, 6, 10, 30],
    2: [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 200],
    3: [10, 20, 30, 40, 60, 80, 120, 160, 240, 320],
}
UTIGSP_GRID = [0.2, 0.1, 0.01, 0.001, 1e-5, 1e-7, 1e-9]
METHODS = ("mip_profiled", "gnies", "utigsp")
FIELDS = [
    "method", "graph", "trial", "penalty", "fdp", "tdp", "d_cpdag",
    "union_target_error", "fit_seconds", "metric_seconds", "mip_gap",
    "status", "error",
]


def _float_key(value):
    if value is None or str(value).strip() == "":
        return ""
    number = float(value)
    return "" if np.isnan(number) else f"{number:.12g}"


def _key(row):
    return (
        row["method"], int(row["graph"]), int(row["trial"]),
        _float_key(row["penalty"]),
    )


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            cleaned = {field: row.get(field, "") for field in FIELDS}
            cleaned["fit_seconds"] = cleaned["fit_seconds"] or row.get(
                "fit_wall_seconds", ""
            )
            rows.append(cleaned)
    return rows


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=_key))
    os.replace(temporary, path)


@contextmanager
def _result_lock(path):
    """Fail fast rather than let overlapping jobs write the same fragment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another comparison runner is already using {path.parent}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _time_limit(seconds, label):
    previous = signal.getsignal(signal.SIGALRM)

    def timeout(_signum, _frame):
        raise TimeoutError(f"{label} exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _run_mip(args, data, penalty, _p):
    import gurobipy as gp

    from MIP import estimate_moral_graph
    from MIP_profiled import optimization
    from src.utils import interventional_cpdag

    gp.setParam("OutputFlag", 1)
    gp.setParam("Threads", args.threads)
    started = time.perf_counter()
    moral = estimate_moral_graph(data[0], ALPHA)
    gamma, targets, gap, _, _ = optimization(
        data,
        moral,
        penalty,
        configuration_limit=CONFIGURATION_LIMIT,
        time_limit=args.time_limit,
    )
    estimated_dag = (np.abs(gamma) > 1e-6).astype(int)
    np.fill_diagonal(estimated_dag, 0)
    targets = np.asarray(targets, dtype=bool)
    return (
        interventional_cpdag(estimated_dag, targets),
        targets.any(axis=0),
        float(gap),
        time.perf_counter() - started,
    )


def _run_gnies(args, data, penalty, p):
    import gnies

    started = time.perf_counter()
    with _time_limit(args.time_limit, "GnIES fit"):
        _, estimated, targets = gnies.fit(data, approach="rank", lmbda=penalty)
    target_union = np.zeros(p, dtype=bool)
    target_union[list(targets)] = True
    return (
        (np.abs(estimated) > 1e-6).astype(int),
        target_union,
        "",
        time.perf_counter() - started,
    )


def _run_utigsp(args, data, penalty, _p):
    from UTIGSP import fit
    from src.utils import interventional_cpdag

    started = time.perf_counter()
    with _time_limit(args.time_limit, "UT-IGSP fit"):
        estimated_dag, targets = fit(
            data,
            alpha=penalty,
            alpha_inv=penalty,
            depth=args.utigsp_depth,
            nruns=args.utigsp_nruns,
            seed=args.seed,
        )
    targets = np.asarray(targets, dtype=bool)
    return (
        interventional_cpdag(estimated_dag, targets),
        targets.any(axis=0),
        "",
        time.perf_counter() - started,
    )


RUNNERS = {
    "mip_profiled": _run_mip,
    "gnies": _run_gnies,
    "utigsp": _run_utigsp,
}


def _metrics(estimated, truth, time_limit):
    from src.utils import cpdag_distance, equivalence_class_fdp_tdp

    started = time.perf_counter()
    with _time_limit(time_limit, "metric evaluation"):
        d_cpdag = cpdag_distance(estimated, truth)
        fdp, tdp = equivalence_class_fdp_tdp(estimated, truth)
    return dict(
        d_cpdag=d_cpdag,
        fdp=fdp,
        tdp=tdp,
        metric_seconds=time.perf_counter() - started,
    )


def _method_grid(method, graph):
    if method == "mip_profiled":
        return MIP_GRIDS[graph]
    if method == "gnies":
        return GNIES_GRIDS[graph]
    return UTIGSP_GRID


def _settings(args, method, graph):
    return (
        args.penalties
        if args.penalties is not None
        else _method_grid(method, graph)
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, required=True, choices=range(1, 11))
    parser.add_argument(
        "--graphs", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mip_profiled", "gnies"],
        choices=METHODS,
    )
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        help="override the primary grid; requires exactly one selected method",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--utigsp-depth", type=int, default=4)
    parser.add_argument("--utigsp-nruns", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--metric-time-limit", type=float, default=3600)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment_results" / "tdp_fdp",
    )
    args = parser.parse_args()
    if min(args.threads, args.time_limit, args.metric_time_limit) <= 0:
        parser.error("threads and time limits must be positive")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.utigsp_depth < 0 or args.utigsp_nruns <= 0:
        parser.error("UT-IGSP depth must be nonnegative and nruns must be positive")
    if args.penalties is not None and len(args.methods) != 1:
        parser.error("--penalties requires exactly one selected method")
    if args.penalties is not None and any(value <= 0 for value in args.penalties):
        parser.error("penalties must be positive")
    return args


def _run(args):
    from MIP import _load
    from src.utils import interventional_cpdag

    result_path = args.output_dir.resolve() / f"trial_{args.trial:02d}.csv"
    rows = {_key(row): row for row in _read_rows(result_path)}
    had_errors = False

    for graph in args.graphs:
        data, _, true_dag, true_targets = _load(DATA_ROOT, graph, args.trial)
        truth = interventional_cpdag(true_dag, true_targets)
        true_target_union = np.asarray(true_targets, dtype=bool).any(axis=0)
        for method in args.methods:
            for penalty in _settings(args, method, graph):
                row = {field: "" for field in FIELDS}
                row.update(
                    method=method,
                    graph=graph,
                    trial=args.trial,
                    penalty=penalty,
                )
                key = _key(row)
                if key in rows and rows[key]["status"].startswith("ok"):
                    print("skip", key, flush=True)
                    continue

                print("run", key, flush=True)
                attempt_started = time.perf_counter()
                fit_completed_at = None
                try:
                    estimated, target_union, gap, fit_seconds = RUNNERS[method](
                        args, data, penalty, len(true_dag)
                    )
                    fit_completed_at = time.perf_counter()
                    row.update(
                        fit_seconds=fit_seconds,
                        mip_gap=gap,
                        union_target_error=int(
                            (target_union != true_target_union).sum()
                        ),
                    )
                    row.update(_metrics(estimated, truth, args.metric_time_limit))
                    row["status"] = (
                        "ok_with_gap"
                        if gap != "" and gap > 1.0001e-4
                        else "ok"
                    )
                except Exception as error:
                    had_errors = True
                    elapsed = time.perf_counter() - attempt_started
                    if fit_completed_at is None:
                        row["fit_seconds"] = elapsed
                    else:
                        row["metric_seconds"] = time.perf_counter() - fit_completed_at
                    row.update(
                        status="error", error=f"{type(error).__name__}: {error}"
                    )
                    print("error", key, row["error"], file=sys.stderr, flush=True)
                rows[key] = row
                _write_rows(result_path, rows.values())
                print(row["status"], key, flush=True)

    print(f"results: {result_path}", flush=True)
    return 1 if had_errors else 0


def main():
    args = _parse_args()
    result_path = args.output_dir.resolve() / f"trial_{args.trial:02d}.csv"
    with _result_lock(result_path.with_suffix(".lock")):
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
