#!/usr/bin/env python3
"""Thin adapter for official known-target IGSP from ``causaldag``."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from src.main_experiment_cli import (
    add_main_data_arguments,
    instance_summary,
    load_cli_instance,
    selected_method_seed,
    wall_time_limit,
)
from src.main_experiment_data import MainExperimentDataError
from UTIGSP import _as_index, _validate_data, _load_hsic_invariance_test


def _load_causaldag_api():
    """Import the official IGSP entry point and its statistical testers."""
    try:
        from causaldag import (
            MemoizedCI_Tester,
            MemoizedInvarianceTester,
            gauss_invariance_suffstat,
            gauss_invariance_test,
            igsp,
            partial_correlation_suffstat,
            partial_correlation_test,
        )
    except Exception as exc:
        raise ImportError(
            "IGSP requires a compatible causaldag installation. "
            "Install `causaldag` and its required dependencies."
        ) from exc

    return (
        MemoizedCI_Tester,
        MemoizedInvarianceTester,
        gauss_invariance_suffstat,
        gauss_invariance_test,
        igsp,
        partial_correlation_suffstat,
        partial_correlation_test,
    )


def _known_target_sets(known_targets, environments, p):
    """Validate the K-by-p oracle matrix and return zero-based target sets."""
    targets = np.asarray(known_targets)
    expected = (environments, p)
    if targets.shape != expected or not np.isin(targets, (0, 1)).all():
        raise ValueError(
            f"known_targets must be binary with shape {expected}; "
            "do not include an observational row"
        )
    return [set(np.flatnonzero(row)) for row in targets]


def fit(
    data,
    known_targets,
    *,
    alpha=1e-5,
    alpha_inv=None,
    invariance_test="gaussian",
    depth=4,
    nruns=10,
    seed=0,
):
    """Fit IGSP with complete target lists and Gaussian or HSIC invariance.

    The Gaussian default preserves the synthetic experiment. Sachs can select
    HSIC explicitly to match the UT-IGSP statistical testing protocol.
    """
    arrays = _validate_data(data)
    p = arrays[0].shape[1]
    target_sets = _known_target_sets(known_targets, len(arrays) - 1, p)
    alpha = float(alpha)
    alpha_inv = alpha if alpha_inv is None else float(alpha_inv)
    if not 0 < alpha < 1 or not 0 < alpha_inv < 1:
        raise ValueError("alpha and alpha_inv must lie strictly between 0 and 1")
    if not isinstance(invariance_test, str) or invariance_test not in ("gaussian", "hsic"):
        raise ValueError("invariance_test must be 'gaussian' or 'hsic'")
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
        igsp,
        partial_correlation_suffstat,
        partial_correlation_test,
    ) = _load_causaldag_api()

    observational = arrays[0]
    interventions = arrays[1:]
    ci_tester = MemoizedCI_Tester(
        partial_correlation_test,
        partial_correlation_suffstat(observational),
        alpha=alpha,
    )
    if invariance_test == "gaussian":
        invariance_fn = gauss_invariance_test
        invariance_suffstat = gauss_invariance_suffstat(observational, interventions)
    else:
        invariance_fn = _load_hsic_invariance_test()
        invariance_suffstat = dict(enumerate(interventions))
        invariance_suffstat["obs_samples"] = observational
    invariance_tester = MemoizedInvarianceTester(
        invariance_fn, invariance_suffstat, alpha=alpha_inv,
    )
    settings = [{"interventions": targets} for targets in target_sets]

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        estimated_dag = igsp(
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
    return adjacency


def main() -> int:
    """Run known-target IGSP on one persisted main-experiment instance."""
    parser = argparse.ArgumentParser(
        description="Run official IGSP with known intervention targets."
    )
    add_main_data_arguments(parser, method_seed=True)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument(
        "--alpha-inv",
        type=float,
        help="invariance-test level; omit to tie it to --alpha",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--nruns", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--metric-time-limit", type=float, default=3600)
    args = parser.parse_args()
    if (
        not np.isfinite([args.time_limit, args.metric_time_limit]).all()
        or args.time_limit <= 0
        or args.metric_time_limit <= 0
    ):
        parser.error("time limits must be positive and finite")

    from src.utils import (
        cpdag_distance,
        equivalence_class_fdp_tdp,
        interventional_cpdag,
    )

    try:
        data, true_dag, true_targets, info = load_cli_instance(args)
    except MainExperimentDataError as error:
        parser.error(str(error))
    method_seed = selected_method_seed(args, info)
    started = time.perf_counter()
    with wall_time_limit(args.time_limit, "IGSP fit"):
        dag = fit(
            data,
            true_targets,
            alpha=args.alpha,
            alpha_inv=args.alpha_inv,
            depth=args.depth,
            nruns=args.nruns,
            seed=method_seed,
        )
    runtime = time.perf_counter() - started
    with wall_time_limit(args.metric_time_limit, "metric evaluation"):
        estimated_icpdag = interventional_cpdag(dag, true_targets)
        true_icpdag = interventional_cpdag(true_dag, true_targets)
        d_cpdag = cpdag_distance(estimated_icpdag, true_icpdag)
        fdp, tdp = equivalence_class_fdp_tdp(estimated_icpdag, true_icpdag)

    effective_alpha_inv = args.alpha if args.alpha_inv is None else args.alpha_inv
    print("Instance:", instance_summary(args, data, true_dag, true_targets, info))
    print(
        "Configuration:",
        {
            "alpha": args.alpha,
            "alpha_inv": effective_alpha_inv,
            "depth": args.depth,
            "nruns": args.nruns,
            "method_seed": method_seed,
            "fit_time_limit": args.time_limit,
            "metric_time_limit": args.metric_time_limit,
            "intervention_knowledge": "known_oracle",
        },
    )
    print("DAG:\n", dag)
    print("I-CPDAG:\n", estimated_icpdag)
    print("Known targets:\n", true_targets)
    print("Runtime:", runtime)
    print("d_cpdag, FDP, TDP:", (d_cpdag, fdp, tdp))
    return 0


__all__ = ["fit", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
