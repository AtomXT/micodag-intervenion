#!/usr/bin/env python3
"""Run the GNIES rank baseline on the synthetic intervention data."""

import argparse
import time
from pathlib import Path

import gnies
import numpy as np

from MIP import _load
from src.utils import cpdag_distance, equivalence_class_fdp_tdp, interventional_cpdag


def compute_errors(icpdag, targets, true_dag, true_targets):
    """Return I-CPDAG distance, union-target error, class FDP, and class TDP."""
    estimate = (np.abs(icpdag) > 1e-6).astype(int)
    true_icpdag = interventional_cpdag(true_dag, true_targets)
    target_mask = np.zeros(len(true_dag), dtype=int)
    target_mask[list(targets)] = 1
    target_error = np.abs(target_mask - true_targets.any(axis=0)).sum()
    fdp, tdp = equivalence_class_fdp_tdp(estimate, true_icpdag)
    return cpdag_distance(estimate, true_icpdag), int(target_error), fdp, tdp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=int, default=2)
    parser.add_argument("--iteration", type=int, default=6)
    parser.add_argument("--lambda-penalty", type=float, default=50.0)
    args = parser.parse_args()

    data, _, true_dag, true_targets = _load(Path("data/SyntheticData"), args.graph, args.iteration)
    start = time.time()
    score, icpdag, targets = gnies.fit(data, approach="rank", lmbda=args.lambda_penalty)
    runtime = time.time() - start
    errors = compute_errors(icpdag, targets, true_dag, true_targets)

    print("I-CPDAG:\n", icpdag.astype(int))
    print("Targets (0-based):", sorted(targets))
    print("Score, runtime:", score, runtime)
    print("d_cpdag, union-target error, FDP, TDP:", errors)
    print("Environment-specific target error: unavailable from GNIES")
