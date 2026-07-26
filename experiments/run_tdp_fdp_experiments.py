#!/usr/bin/env python3
"""Run checkpointed MIP-profiled and GnIES FDP/TDP experiments."""

import argparse
import csv
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
MIP_GRID = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.32, 0.64, 1.28]
MIP_GRIDS = {graph: MIP_GRID for graph in (1, 2, 3)}
GNIES_GRIDS = {
    1: [0.005, 0.01, 0.03, 0.1, 0.3, 1, 3, 6, 10, 30, 100, 300],
    2: [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 200, 300],
    3: [10, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640],
}
FIELDS = [
    "method", "graph", "trial", "penalty", "fdp", "tdp", "d_cpdag",
    "union_target_error", "fit_seconds", "metric_seconds", "mip_gap",
    "status", "error",
]


def _key(row):
    return row["method"], int(row["graph"]), int(row["trial"]), f"{float(row['penalty']):.12g}"


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            cleaned = {field: row.get(field, "") for field in FIELDS}
            cleaned["fit_seconds"] = cleaned["fit_seconds"] or row.get("fit_wall_seconds", "")
            rows.append(cleaned)
    return rows


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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
        data, moral, penalty,
        configuration_limit=CONFIGURATION_LIMIT,
        time_limit=args.time_limit,
    )
    estimated_dag = (np.abs(gamma) > 1e-6).astype(int)
    np.fill_diagonal(estimated_dag, 0)
    targets = np.asarray(targets, dtype=bool)
    return (
        interventional_cpdag(estimated_dag, targets), targets.any(axis=0), float(gap),
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
        (np.abs(estimated) > 1e-6).astype(int), target_union, "",
        time.perf_counter() - started,
    )


def _metrics(estimated, truth, time_limit):
    from src.utils import cpdag_distance, equivalence_class_fdp_tdp
    started = time.perf_counter()
    with _time_limit(time_limit, "metric evaluation"):
        d_cpdag = cpdag_distance(estimated, truth)
        fdp, tdp = equivalence_class_fdp_tdp(estimated, truth)
    return dict(d_cpdag=d_cpdag, fdp=fdp, tdp=tdp,
                metric_seconds=time.perf_counter() - started)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, required=True, choices=range(1, 11))
    parser.add_argument("--graphs", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument(
        "--methods", nargs="+", default=["mip_profiled", "gnies"],
        choices=["mip_profiled", "gnies"]
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--metric-time-limit", type=float, default=3600)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "experiment_results" / "tdp_fdp")
    args = parser.parse_args()
    if min(args.threads, args.time_limit, args.metric_time_limit) <= 0:
        parser.error("threads and time limits must be positive")
    return args


def main():
    args = _parse_args()
    from MIP import _load
    from src.utils import interventional_cpdag
    result_path = args.output_dir.resolve() / f"trial_{args.trial:02d}.csv"
    rows = {_key(row): row for row in _read_rows(result_path)}

    for graph in args.graphs:
        data, _, true_dag, true_targets = _load(DATA_ROOT, graph, args.trial)
        truth = interventional_cpdag(true_dag, true_targets)
        true_target_union = np.asarray(true_targets, dtype=bool).any(axis=0)
        for method in args.methods:
            grid = MIP_GRIDS[graph] if method == "mip_profiled" else GNIES_GRIDS[graph]
            for penalty in grid:
                key = method, graph, args.trial, f"{penalty:.12g}"
                if key in rows and rows[key]["status"].startswith("ok"):
                    print("skip", key, flush=True)
                    continue

                print("run", key, flush=True)
                row = {field: "" for field in FIELDS}
                row.update(method=method, graph=graph, trial=args.trial, penalty=penalty)
                attempt_started = time.perf_counter()
                try:
                    runner = _run_mip if method == "mip_profiled" else _run_gnies
                    estimated, target_union, gap, fit_seconds = runner(
                        args, data, penalty, len(true_dag)
                    )
                    row.update(
                        fit_seconds=fit_seconds,
                        mip_gap=gap,
                        union_target_error=int((target_union != true_target_union).sum()),
                    )
                    row.update(_metrics(estimated, truth, args.metric_time_limit))
                    row["status"] = "ok_with_gap" if gap != "" and gap > 1.0001e-4 else "ok"
                except Exception as error:
                    elapsed = time.perf_counter() - attempt_started
                    if row["fit_seconds"] == "":
                        row["fit_seconds"] = elapsed
                    else:
                        row["metric_seconds"] = elapsed - row["fit_seconds"]
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    print("error", key, row["error"], file=sys.stderr, flush=True)
                rows[key] = row
                _write_rows(result_path, list(rows.values()))
                print(row["status"], key, flush=True)

    print(f"results: {result_path}", flush=True)


if __name__ == "__main__":
    main()
