#!/usr/bin/env python3
"""Adapter for Unknown-Target Interventional GSP (UT-IGSP)."""

from __future__ import annotations

import argparse
import operator
import random
import time
from pathlib import Path

import numpy as np


def _load_causaldag_api():
    """Import the optional UT-IGSP implementation with a useful failure message."""
    try:
        from causaldag import (
            MemoizedCI_Tester,
            MemoizedInvarianceTester,
            gauss_invariance_suffstat,
            gauss_invariance_test,
            partial_correlation_suffstat,
            partial_correlation_test,
            unknown_target_igsp,
        )
    except Exception as exc:
        raise ImportError(
            "UT-IGSP requires a compatible causaldag installation. "
            "Using Python 3.9, install `causaldag==0.1a163` and "
            "`pgmpy==0.1.25`."
        ) from exc

    return (
        MemoizedCI_Tester,
        MemoizedInvarianceTester,
        gauss_invariance_suffstat,
        gauss_invariance_test,
        partial_correlation_suffstat,
        partial_correlation_test,
        unknown_target_igsp,
    )


def _validate_data(data):
    """Return finite, consistently shaped arrays in environment order."""
    try:
        datasets = list(data)
    except TypeError as exc:
        raise ValueError("data must be an iterable of sample matrices") from exc
    if len(datasets) < 2:
        raise ValueError(
            "data must contain observational data followed by at least one "
            "interventional environment"
        )

    arrays = []
    for environment, dataset in enumerate(datasets):
        values = dataset.values if hasattr(dataset, "values") else dataset
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"environment {environment} must contain numeric data"
            ) from exc
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(
                f"environment {environment} must be a nonempty two-dimensional array"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"environment {environment} contains nonfinite values")
        arrays.append(array)

    p = arrays[0].shape[1]
    if any(array.shape[1] != p for array in arrays[1:]):
        raise ValueError("all environments must contain the same ordered variables")
    return arrays


def _as_index(value, name, *, minimum=0):
    """Return an integer parameter after rejecting booleans and invalid bounds."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _target_matrix(target_sets, environments, p):
    """Convert causaldag's environment-indexed target sets to a binary matrix."""
    if len(target_sets) != environments:
        raise RuntimeError(
            "causaldag returned a target list with the wrong number of environments"
        )

    targets = np.zeros((environments, p), dtype=int)
    for environment, target_set in enumerate(target_sets):
        for node in target_set:
            node = _as_index(node, "estimated target node")
            if node >= p:
                raise RuntimeError(
                    f"causaldag returned invalid target node {node} for p={p}"
                )
            targets[environment, node] = 1
    return targets


def fit(
    data,
    *,
    alpha=1e-5,
    alpha_inv=None,
    depth=4,
    nruns=10,
    seed=0,
):
    """Fit UT-IGSP with all intervention targets treated as unknown.

    Parameters
    ----------
    data:
        Sample matrices in environment order. ``data[0]`` is observational;
        every remaining matrix is one interventional environment. Matrices
        must use the same variable order.
    alpha:
        Significance level for observational conditional-independence tests.
    alpha_inv:
        Significance level for conditional-invariance tests. If omitted, it
        is tied to ``alpha``.
    depth, nruns:
        UT-IGSP search depth and number of initial runs. Their defaults match
        the published simulation implementation.
    seed:
        Seed for both Python and NumPy global random generators. Their prior
        states are restored before this function returns.

    Returns
    -------
    (dag, targets):
        ``dag`` is a ``p``-by-``p`` binary adjacency matrix with ``dag[i, j]``
        indicating ``i -> j``. ``targets`` is a ``K``-by-``p`` binary matrix,
        with one row per interventional environment and no observational row.
    """
    arrays = _validate_data(data)
    p = arrays[0].shape[1]
    alpha = float(alpha)
    alpha_inv = alpha if alpha_inv is None else float(alpha_inv)
    if not 0 < alpha < 1 or not 0 < alpha_inv < 1:
        raise ValueError("alpha and alpha_inv must lie strictly between 0 and 1")
    depth = _as_index(depth, "depth")
    nruns = _as_index(nruns, "nruns", minimum=1)
    seed = _as_index(seed, "seed")
    if seed > 2**32 - 1:
        raise ValueError("seed must be at most 2**32 - 1")

    (
        MemoizedCI_Tester,
        MemoizedInvarianceTester,
        gauss_invariance_suffstat,
        gauss_invariance_test,
        partial_correlation_suffstat,
        partial_correlation_test,
        unknown_target_igsp,
    ) = _load_causaldag_api()

    observational = arrays[0]
    interventions = arrays[1:]
    ci_suffstat = partial_correlation_suffstat(observational)
    invariance_suffstat = gauss_invariance_suffstat(
        observational, interventions
    )
    ci_tester = MemoizedCI_Tester(
        partial_correlation_test, ci_suffstat, alpha=alpha
    )
    invariance_tester = MemoizedInvarianceTester(
        gauss_invariance_test, invariance_suffstat, alpha=alpha_inv
    )
    settings = [dict(known_interventions=set()) for _ in interventions]

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        estimated_dag, estimated_target_sets = unknown_target_igsp(
            settings,
            set(range(p)),
            ci_tester,
            invariance_tester,
            depth=depth,
            nruns=nruns,
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    adjacency = np.asarray(
        estimated_dag.to_amat(node_list=list(range(p)))[0]
    )
    if adjacency.shape != (p, p):
        raise RuntimeError("causaldag returned an adjacency matrix with the wrong shape")
    adjacency = (adjacency != 0).astype(int)
    np.fill_diagonal(adjacency, 0)
    targets = _target_matrix(estimated_target_sets, len(interventions), p)
    return adjacency, targets


def main():
    """Run UT-IGSP on one repository synthetic dataset and print diagnostics."""
    parser = argparse.ArgumentParser(
        description="Run UT-IGSP with unknown intervention targets."
    )
    parser.add_argument("--graph", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=5e-3)
    parser.add_argument(
        "--alpha-inv",
        type=float,
        help="invariance-test level; omit to tie it to --alpha",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--nruns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "SyntheticData",
    )
    args = parser.parse_args()

    from MIP import _load
    from src.utils import compute_errors, interventional_cpdag

    data, _, true_dag, true_targets = _load(
        args.data_root, args.graph, args.iteration
    )
    started = time.perf_counter()
    dag, targets = fit(
        data,
        alpha=args.alpha,
        alpha_inv=args.alpha_inv,
        depth=args.depth,
        nruns=args.nruns,
        seed=args.seed,
    )
    runtime = time.perf_counter() - started
    estimated_icpdag = interventional_cpdag(dag, targets)
    errors = compute_errors(dag, targets, true_dag, true_targets)

    print("DAG:\n", dag)
    print("I-CPDAG:\n", estimated_icpdag)
    print("Targets:\n", targets)
    print("Runtime:", runtime)
    print("d_cpdag, environment target error, FDP, TDP:", errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
