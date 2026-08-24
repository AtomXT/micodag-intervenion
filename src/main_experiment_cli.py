"""Shared CLI helpers for directly running methods on main-experiment data."""

from __future__ import annotations

import signal
from contextlib import contextmanager
from math import isfinite, log, sqrt
from pathlib import Path

from src.main_experiment_data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MASTER_SEED,
    DEFAULT_NUM_REPLICATES,
    EDGE_MULTIPLIERS,
    P_VALUES,
    load_main_experiment_instance,
)


def add_main_data_arguments(parser, *, method_seed: bool = False) -> None:
    """Add the canonical persisted-instance selectors to an argument parser."""
    parser.add_argument("--p", type=int, choices=P_VALUES, default=min(P_VALUES))
    parser.add_argument(
        "--edge-multiplier",
        "--e",
        dest="edge_multiplier",
        type=int,
        choices=EDGE_MULTIPLIERS,
        default=min(EDGE_MULTIPLIERS),
        help="edge multiplier e; expected directed-edge count is e*p",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        choices=range(1, DEFAULT_NUM_REPLICATES + 1),
        default=1,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"persisted main-experiment suite (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help=f"dataset-suite master seed (default: {DEFAULT_MASTER_SEED})",
    )
    if method_seed:
        parser.add_argument(
            "--method-seed",
            type=int,
            help="override the stored instance seed used by the fitting method",
        )


def load_cli_instance(args):
    """Load and validate the persisted instance selected by shared CLI arguments."""
    return load_main_experiment_instance(
        Path(args.data_root).expanduser().resolve(),
        args.p,
        args.edge_multiplier,
        args.replicate,
        master_seed=args.seed,
    )


def selected_method_seed(args, instance_info) -> int:
    override = getattr(args, "method_seed", None)
    return int(instance_info["seed"] if override is None else override)


def main_screen_alpha(data, constant: float = 5.0) -> float:
    """Return c*sqrt(log(p)/n_obs), the main PS-MIP screening scale."""
    if not isfinite(constant) or constant <= 0:
        raise ValueError("screen constant must be positive and finite")
    p = int(data[0].shape[1])
    return float(constant * sqrt(log(p) / len(data[0])))


def primary_mip_penalty(data) -> float:
    """Return the factor-one main PS-MIP penalty log(N)/N."""
    total = sum(len(values) for values in data)
    return float(log(total) / total)


def primary_score_penalty(data) -> float:
    """Return the factor-one native GIES/GnIES BIC penalty 0.5*log(N)."""
    return float(0.5 * log(sum(len(values) for values in data)))


def instance_summary(args, data, true_dag, true_targets, instance_info) -> str:
    return (
        f"p={args.p}, e={args.edge_multiplier}, replicate={args.replicate}, "
        f"master_seed={args.seed}, instance_seed={instance_info['seed']}, samples="
        f"{[len(values) for values in data]}, edges={int(true_dag.sum())}, "
        f"target_entries={int(true_targets.sum())}"
    )


@contextmanager
def wall_time_limit(seconds: float, label: str):
    """Bound a direct in-process fit or metric call on the POSIX main platform."""
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError("time limit must be positive and finite")
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


__all__ = [
    "add_main_data_arguments",
    "instance_summary",
    "load_cli_instance",
    "main_screen_alpha",
    "primary_mip_penalty",
    "primary_score_penalty",
    "selected_method_seed",
    "wall_time_limit",
]
