"""Shared graph-evaluation utilities for the active experiment pipeline."""

from __future__ import annotations

import warnings

import numpy as np


def cpdag_distance(estimated, truth, threshold=1e-6):
    """Return the entrywise L1 distance between two CPDAG adjacencies."""
    estimated = np.asarray(estimated)
    truth = np.asarray(truth)

    if estimated.shape != truth.shape or estimated.ndim != 2:
        raise ValueError("estimated and truth must be equally sized matrices")
    if estimated.shape[0] != estimated.shape[1]:
        raise ValueError("adjacency matrices must be square")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if not np.isfinite(estimated).all() or not np.isfinite(truth).all():
        raise ValueError("adjacency matrices must contain only finite values")

    estimated_edges = (np.abs(estimated) > threshold).astype(int)
    true_edges = (np.abs(truth) > threshold).astype(int)
    if np.any(np.diag(estimated_edges)) or np.any(np.diag(true_edges)):
        raise ValueError("adjacency matrices must have zero diagonals")

    return int(np.abs(estimated_edges - true_edges).sum())


def interventional_cpdag(dag, targets, threshold=1e-6):
    """Return the I-CPDAG adjacency for a DAG and environment target matrix."""
    import causaldag as cd

    dag = np.asarray(dag)
    targets = np.asarray(targets)
    if dag.ndim != 2 or dag.shape[0] != dag.shape[1]:
        raise ValueError("dag must be a square adjacency matrix")
    if targets.ndim != 2 or targets.shape[1] != len(dag):
        raise ValueError("targets must have shape (environments, variables)")

    graph = cd.DAG.from_amat((np.abs(dag) > threshold).astype(int))
    target_sets = [set(np.flatnonzero(row)) for row in targets]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return graph.interventional_cpdag(
            target_sets, cpdag=graph.cpdag()
        ).to_amat()[0]


def _equivalence_class(pdag, threshold=1e-6):
    """Return directed-edge sets for all DAG extensions of an (I-)CPDAG."""
    import causaldag as cd

    pdag = np.asarray(pdag)
    if pdag.ndim != 2 or pdag.shape[0] != pdag.shape[1]:
        raise ValueError("pdag must be a square adjacency matrix")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if not np.isfinite(pdag).all():
        raise ValueError("pdag must contain only finite values")

    adjacency = (np.abs(pdag) > threshold).astype(int)
    if np.any(np.diag(adjacency)):
        raise ValueError("pdag must have a zero diagonal")
    dags = cd.PDAG.from_amat(adjacency).all_dags()
    if not dags:
        raise ValueError("pdag has no consistent DAG extension")
    return dags


def _max_min_edge_error(first_class, second_class):
    """Return the max-min share of first-class edges absent from the second."""
    maximum = 0.0
    for first in first_class:
        if not first:
            distance = 0.0
        else:
            distance = min(
                len(first - second) / len(first) for second in second_class
            )
        maximum = max(maximum, distance)
    return maximum


def equivalence_class_fdp_tdp(estimated_pdag, true_pdag, threshold=1e-6):
    """Return the equivalence-class FDP and TDP from Taeb et al. (2024)."""
    estimated_class = _equivalence_class(estimated_pdag, threshold)
    true_class = _equivalence_class(true_pdag, threshold)
    fdp = _max_min_edge_error(estimated_class, true_class)
    false_negative_proportion = _max_min_edge_error(
        true_class, estimated_class
    )
    return float(fdp), float(1.0 - false_negative_proportion)


def compute_errors(gamma, targets, true_dag, true_targets):
    """Return I-CPDAG distance, target error, class FDP, and class TDP."""
    estimated_dag = (np.abs(gamma) > 1e-6).astype(int)
    np.fill_diagonal(estimated_dag, 0)
    estimated_targets = np.asarray(targets, dtype=int)
    true_targets = np.asarray(true_targets, dtype=int)
    if estimated_targets.shape != true_targets.shape:
        raise ValueError("estimated and true targets must have the same shape")

    estimated_icpdag = interventional_cpdag(estimated_dag, estimated_targets)
    true_icpdag = interventional_cpdag(true_dag, true_targets)
    d_cpdag = cpdag_distance(estimated_icpdag, true_icpdag)
    fdp, tdp = equivalence_class_fdp_tdp(estimated_icpdag, true_icpdag)
    target_error = int(np.abs(estimated_targets - true_targets).sum())
    return d_cpdag, target_error, fdp, tdp


__all__ = [
    "compute_errors",
    "cpdag_distance",
    "equivalence_class_fdp_tdp",
    "interventional_cpdag",
]
