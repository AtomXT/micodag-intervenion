#!/usr/bin/env python3
"""Fully profiled parent-set MIP for unknown intervention targets."""

from __future__ import annotations

import argparse
from itertools import combinations
from math import comb

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from MIP import _edges, _moments, estimate_moral_graph
from src.main_experiment_cli import (
    add_main_data_arguments,
    instance_summary,
    load_cli_instance,
    main_screen_alpha,
    primary_mip_penalty,
    wall_time_limit,
)
from src.main_experiment_data import MainExperimentDataError
from src.utils import compute_errors


def _best_local(
    j,
    parents,
    sigma,
    weights,
    l,
    delta_penalties,
    status,
    coefficient_bound,
    lower,
    upper,
):
    """Profile Gamma and the intervention pattern for one parent set."""
    environments = len(sigma)
    candidates = []
    for mask in range(1 << (environments - 1)):
        bits = np.array([(mask >> e) & 1 for e in range(environments - 1)])
        if np.any((status >= 0) & (bits != status)):
            continue
        nonintervened = [0] + [e for e in range(1, environments) if not bits[e - 1]]
        a = float(weights[nonintervened].sum())
        indices = list(parents) + [j]
        Q = sum(weights[e] * sigma[e][np.ix_(indices, indices)] for e in nonintervened)
        if parents:
            slope = np.linalg.solve(Q[:-1, :-1], Q[:-1, -1])
            residual = float(Q[-1, -1] - Q[-1, :-1] @ slope)
        else:
            slope, residual = np.empty(0), float(Q[-1, -1])
        if residual <= 0:
            raise ValueError("profiled covariance must be positive definite")

        diagonal = np.sqrt(a / residual)
        coefficients = -slope * diagonal
        score = a * (1 + np.log(residual / a)) + l * len(parents)
        score += sum(
            weights[e] * (1 + np.log(sigma[e][j, j]))
            + delta_penalties[e - 1]
            for e in range(1, environments) if bits[e - 1]
        )
        intervened_diagonals = [
            1 / np.sqrt(sigma[e][j, j])
            for e in range(1, environments) if bits[e - 1]
        ]
        valid = (
            lower <= diagonal <= upper
            and np.all(np.abs(coefficients) <= coefficient_bound)
            and all(lower <= value <= upper for value in intervened_diagonals)
        )
        candidates.append((score, mask, diagonal, coefficients, valid))

    valid = [item for item in candidates if item[-1]]
    if not valid:
        raise ValueError(f"no bounded local optimum for node {j}, parents {parents}")
    best = min(valid, key=lambda item: item[0])
    if min(item[0] for item in candidates) < best[0] - 1e-10:
        raise ValueError("an active Gamma bound requires a bounded local optimizer")
    return best[:-1]


def optimization(
    data,
    moral,
    l,
    *,
    l_delta=None,
    coefficient_bound=10.0,
    gamma_lower=1e-5,
    gamma_upper=10.0,
    target_status=None,
    max_parents=None,
    configuration_limit=1_000_000,
    time_limit=None,
    return_metadata=False,
):
    """Return Gamma, targets, MIP gap, objective, and runtime.

    If ``l_delta`` is omitted, environment ``e`` uses the intervention
    penalty ``l * weights[e]``. A supplied scalar applies the same penalty
    to every interventional environment; a vector supplies one penalty per
    interventional environment. ``target_status`` has one row per
    non-observational environment; values 0/1 fix an intervention indicator
    and -1 leaves it unknown. If ``return_metadata`` is true, a sixth return
    value contains the solver status and bound without changing legacy calls.
    """
    if l < 0 or coefficient_bound <= 0 or not 0 < gamma_lower < gamma_upper:
        raise ValueError("penalties and Gamma bounds are invalid")

    data, sigma, weights = _moments(data)
    environments, p = len(data), data[0].shape[1]
    if l_delta is None:
        delta_penalties = l * weights[1:]
    else:
        supplied_penalties = np.asarray(l_delta, dtype=float)
        if supplied_penalties.ndim == 0:
            delta_penalties = np.full(environments - 1, float(supplied_penalties))
        elif supplied_penalties.shape == (environments - 1,):
            delta_penalties = supplied_penalties
        else:
            raise ValueError(
                "l_delta must be a scalar or have one value per interventional environment"
            )
    if not np.isfinite(delta_penalties).all() or np.any(delta_penalties < 0):
        raise ValueError("penalties and Gamma bounds are invalid")

    candidates = _edges(moral, p)
    if any(np.any(np.diag(s) <= 0) for s in sigma):
        raise ValueError("every variable must have positive empirical variance")
    if target_status is None:
        status = -np.ones((environments - 1, p), dtype=int)
    else:
        status = np.asarray(target_status)
        if status.shape != (environments - 1, p) or not np.isin(status, (-1, 0, 1)).all():
            raise ValueError("target_status must have shape (environments - 1, p)")
        status = status.astype(int)
    if max_parents is not None and max_parents < 0:
        raise ValueError("max_parents must be nonnegative")

    possible = {j: sorted(i for i in range(p) if (i, j) in candidates) for j in range(p)}
    sizes = {
        j: range(min(len(possible[j]), max_parents if max_parents is not None else p) + 1)
        for j in range(p)
    }
    count = sum(comb(len(possible[j]), size) for j in range(p) for size in sizes[j])
    if configuration_limit is not None and count > configuration_limit:
        raise ValueError(f"{count:,} parent sets exceed configuration_limit")

    configs, by_node = [], [[] for _ in range(p)]
    for j in range(p):
        for size in sizes[j]:
            for parents in combinations(possible[j], size):
                score, mask, diagonal, coefficients = _best_local(
                    j, parents, sigma, weights, l, delta_penalties, status[:, j],
                    coefficient_bound, gamma_lower, gamma_upper,
                )
                by_node[j].append(len(configs))
                configs.append((j, parents, mask, score, diagonal, coefficients))

    model = gp.Model("profiled_interventional_dag_mip")
    choose = model.addVars(len(configs), vtype=GRB.BINARY, name="choose")
    order = model.addVars(p, lb=1, ub=p, name="order")
    model.addConstrs(
        (gp.quicksum(choose[r] for r in by_node[j]) == 1 for j in range(p)),
        name="one_parent_set",
    )
    used_arcs = {(i, c[0]) for c in configs for i in c[1]}
    for i, j in used_arcs:
        selected = gp.quicksum(choose[r] for r in by_node[j] if i in configs[r][1])
        model.addConstr(1 - p + p * selected <= order[j] - order[i])
    model.setObjective(
        gp.quicksum(configs[r][3] * choose[r] for r in range(len(configs))),
        GRB.MINIMIZE,
    )
    model.Params.MIPGap = 1e-4
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi found no feasible solution; status {model.Status}")

    gamma = np.zeros((p, p))
    targets = np.zeros((environments - 1, p), dtype=int)
    for j in range(p):
        config = configs[max(by_node[j], key=lambda r: choose[r].X)]
        _, parents, mask, _, diagonal, coefficients = config
        gamma[j, j] = diagonal
        gamma[list(parents), j] = coefficients
        targets[:, j] = [(mask >> e) & 1 for e in range(environments - 1)]
    result = gamma, targets.tolist(), model.MIPGap, model.ObjVal, model.Runtime
    if not return_metadata:
        return result

    status_name = next(
        (
            name.lower()
            for name in (
                "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
                "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
                "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
                "INPROGRESS", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT",
            )
            if getattr(GRB, name, None) == model.Status
        ),
        f"status_{model.Status}",
    )
    metadata = {
        "solver_status_code": int(model.Status),
        "solver_status": status_name,
        "optimal": bool(model.Status == GRB.OPTIMAL),
        "solution_count": int(model.SolCount),
        "objective_bound": float(model.ObjBound),
        "node_count": float(model.NodeCount),
    }
    return (*result, metadata)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run profiled PS-MIP on one persisted main-experiment instance."
    )
    add_main_data_arguments(parser)
    parser.add_argument(
        "--lambda-graph",
        type=float,
        help="graph penalty; omit for the primary log(N)/N setting",
    )
    parser.add_argument(
        "--lambda-delta",
        type=float,
        help="common target penalty; omit to use the primary graph penalty",
    )
    parser.add_argument("--screen-constant", type=float, default=5.0)
    parser.add_argument("--max-parents", type=int)
    parser.add_argument("--configuration-limit", type=int, default=1_100_000)
    parser.add_argument(
        "--known-targets",
        action="store_true",
        help="run the main experiment's oracle target-fixation variant",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--metric-time-limit", type=float, default=3600)
    args = parser.parse_args()
    numeric_values = [
        args.time_limit,
        args.metric_time_limit,
        args.screen_constant,
        *([] if args.lambda_graph is None else [args.lambda_graph]),
        *([] if args.lambda_delta is None else [args.lambda_delta]),
    ]
    if (
        args.threads < 1
        or not np.isfinite(numeric_values).all()
        or args.time_limit <= 0
        or args.metric_time_limit <= 0
        or args.screen_constant <= 0
        or args.configuration_limit <= 0
        or (args.lambda_graph is not None and args.lambda_graph < 0)
        or (args.lambda_delta is not None and args.lambda_delta < 0)
    ):
        parser.error(
            "threads, limits, and screening controls must be positive and finite; "
            "penalties must be finite and nonnegative"
        )
    try:
        data, true_dag, true_targets, info = load_cli_instance(args)
    except MainExperimentDataError as error:
        parser.error(str(error))
    graph_penalty = (
        primary_mip_penalty(data)
        if args.lambda_graph is None
        else args.lambda_graph
    )
    target_penalty = (
        graph_penalty if args.lambda_delta is None else args.lambda_delta
    )
    alpha = main_screen_alpha(data, args.screen_constant)
    moral = estimate_moral_graph(data[0], alpha)
    gp.setParam("Threads", args.threads)
    result = optimization(
        data,
        moral,
        graph_penalty,
        l_delta=target_penalty,
        target_status=true_targets if args.known_targets else None,
        max_parents=args.max_parents,
        configuration_limit=args.configuration_limit,
        time_limit=args.time_limit,
        return_metadata=True,
    )
    with wall_time_limit(args.metric_time_limit, "metric evaluation"):
        errors = compute_errors(result[0], result[1], true_dag, true_targets)
    print("Instance:", instance_summary(args, data, true_dag, true_targets, info))
    print(
        "Configuration:",
        {"screen_constant": args.screen_constant, "screen_alpha": alpha,
         "graph_penalty": graph_penalty, "target_penalty": target_penalty,
         "known_targets": args.known_targets, "threads": args.threads,
         "fit_time_limit": args.time_limit,
         "metric_time_limit": args.metric_time_limit},
    )
    print("Gamma:\n", result[0])
    print("Targets:", result[1])
    print("MIP gap, objective, runtime:", result[2:5])
    print("Solver metadata:", result[5])
    print("d_cpdag, environment target error, FDP, TDP:", errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
