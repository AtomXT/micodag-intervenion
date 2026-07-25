#!/usr/bin/env python3
"""Run the GNIES rank baseline on the synthetic intervention data."""

import argparse
import time
from pathlib import Path

import gnies
import numpy as np

from MIP import _load
from src.utils import cpdag_distance, interventional_cpdag


def compute_errors(icpdag, targets, true_dag, true_targets):
    """Return I-CPDAG distance, union-target error, skeleton TPR, and FPR."""
    estimate = (np.abs(icpdag) > 1e-6).astype(int)
    true_icpdag = interventional_cpdag(true_dag, true_targets)
    estimated_skeleton = ((estimate + estimate.T) > 0).astype(int)
    true_skeleton = ((true_dag + true_dag.T) > 0).astype(int)
    tp, positives = (estimated_skeleton * true_skeleton).sum(), true_skeleton.sum()
    target_mask = np.zeros(len(true_dag), dtype=int)
    target_mask[list(targets)] = 1
    target_error = np.abs(target_mask - true_targets.any(axis=0)).sum()
    fpr = (estimated_skeleton.sum() - tp) / (true_dag.size - len(true_dag) - positives)
    return cpdag_distance(estimate, true_icpdag), int(target_error), float(tp / positives), float(fpr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=6)
    parser.add_argument("--lambda-penalty", type=float, default=8.0)
    args = parser.parse_args()

    data, _, true_dag, true_targets = _load(Path("data/SyntheticData"), args.graph, args.iteration)
    start = time.time()
    score, icpdag, targets = gnies.fit(data, approach="rank", lmbda=args.lambda_penalty)
    runtime = time.time() - start
    errors = compute_errors(icpdag, targets, true_dag, true_targets)

    print("I-CPDAG:\n", icpdag.astype(int))
    print("Targets (0-based):", sorted(targets))
    print("Score, runtime:", score, runtime)
    print("d_cpdag, union-target error, TPR, FPR:", errors)
    print("Environment-specific target error: unavailable from GNIES")
