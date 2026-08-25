#!/usr/bin/env python3
"""Equation (4.4): DAG learning with unknown intervention targets."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.covariance import GraphicalLasso
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

try:
    import gurobipy as gp
    from gurobipy import GRB, nlfunc
except ImportError:
    gp = GRB = nlfunc = None


def _moments(data):
    """Return arrays, empirical second moments, and sample-size weights."""
    arrays = [np.asarray(x.values if hasattr(x, "values") else x, dtype=float) for x in data]
    if not arrays or any(x.ndim != 2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("data must be a list of finite two-dimensional arrays")
    p = arrays[0].shape[1]
    if any(x.shape[0] == 0 or x.shape[1] != p for x in arrays):
        raise ValueError("all environments must be nonempty and have the same variables")
    n = np.array([len(x) for x in arrays])
    sigma = [np.einsum("ni,nj->ij", x, x) / len(x) for x in arrays]
    return arrays, sigma, n / n.sum()


def estimate_moral_graph(observational_data, alpha):
    """Estimate an undirected superstructure from observational data."""
    x = np.asarray(
        observational_data.values
        if hasattr(observational_data, "values")
        else observational_data,
        dtype=float,
    )
    if x.ndim != 2 or not np.isfinite(x).all() or alpha <= 0:
        raise ValueError("observational_data must be finite and alpha must be positive")
    scale = x.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("graphical lasso requires nonconstant variables")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        precision = GraphicalLasso(alpha=alpha).fit(
            (x - x.mean(axis=0)) / scale
        ).precision_
    moral = np.abs(precision) > 1e-8
    moral |= moral.T
    np.fill_diagonal(moral, False)
    return moral


def _edges(moral, p):
    """Convert a one-based edge list or adjacency matrix to directed candidates."""
    if moral is None:
        return {(i, j) for i in range(p) for j in range(p) if i != j}
    a = np.asarray(moral.values if hasattr(moral, "values") else moral)
    if a.shape == (p, p):
        a = np.argwhere((a != 0) | (a.T != 0))
    else:
        if a.ndim != 2 or a.shape[1] < 2:
            raise ValueError("moral must be an adjacency matrix or two-column edge list")
        a = a[:, :2].astype(int) - 1
    if a.size and (a.min() < 0 or a.max() >= p):
        raise ValueError("moral contains an invalid node label")
    return {(int(i), int(j)) for i, j in a if i != j} | {(int(j), int(i)) for i, j in a if i != j}


def compute_cost_big_m(sigma, weights, edges, M, lower, upper, lambda_delta):
    """Bound |C_off - C_on| over the bounded feasible region."""
    p, bounds = sigma[0].shape[0], np.zeros((len(sigma) - 1, sigma[0].shape[0]))
    degree = [sum((i, j) in edges for i in range(p) if i != j) for j in range(p)]
    for e in range(1, len(sigma)):
        eigenvalues = np.linalg.eigvalsh(sigma[e])
        lo_eig, hi_eig = min(eigenvalues[0], 0), max(eigenvalues[-1], 0)
        for j in range(p):
            if sigma[e][j, j] <= 0:
                raise ValueError(f"environment {e}, variable {j} has zero variance")
            norm = upper**2 + degree[j] * M**2
            lo = weights[e] * (-2 * np.log(upper) + lo_eig * norm)
            hi = weights[e] * (-2 * np.log(lower) + hi_eig * norm)
            on = weights[e] * (1 + np.log(sigma[e][j, j])) + lambda_delta
            bounds[e - 1, j] = max(abs(lo - on), abs(hi - on)) + 1e-8
    return bounds


def optimization(
    data,
    moral,
    l,
    *,
    l_delta=None,
    coefficient_bound=10.0,
    gamma_lower=1e-5,
    gamma_upper=10.0,
    cost_big_m=None,
    target_status=None,
    time_limit=None,
):
    """Return Gamma, intervention targets, MIP gap, objective, and runtime."""
    if gp is None:
        raise ModuleNotFoundError("Install gurobipy and configure a Gurobi license")
    l_delta = l if l_delta is None else l_delta
    if min(l, l_delta) < 0 or not 0 < gamma_lower < gamma_upper:
        raise ValueError("penalties and Gamma bounds are invalid")

    data, sigma, weights = _moments(data)
    environments, p = len(data), data[0].shape[1]
    candidates = _edges(moral, p)
    arcs = [(i, j) for i in range(p) for j in range(p) if i != j]
    selectors = [(e, j) for e in range(1, environments) for j in range(p)]
    B = (
        compute_cost_big_m(
            sigma, weights, candidates, coefficient_bound,
            gamma_lower, gamma_upper, l_delta,
        )
        if cost_big_m is None
        else np.broadcast_to(cost_big_m, (environments - 1, p))
    )

    model = gp.Model("interventional_dag_mip")
    Gamma = model.addVars(p, p, lb=-coefficient_bound, ub=coefficient_bound, name="Gamma")
    g = model.addVars(arcs, vtype=GRB.BINARY, name="g")
    psi = model.addVars(p, lb=1, ub=p, name="psi")
    log_gamma = model.addVars(
        p, lb=np.log(gamma_lower), ub=np.log(gamma_upper), name="log_Gamma"
    )
    delta = model.addVars(selectors, vtype=GRB.BINARY, name="delta")
    theta = model.addVars(selectors, lb=-GRB.INFINITY, name="theta")

    for j in range(p):
        Gamma[j, j].LB, Gamma[j, j].UB = gamma_lower, gamma_upper
        model.addGenConstrNL(log_gamma[j], nlfunc.log(Gamma[j, j]))
    for i, j in arcs:
        if (i, j) in candidates:
            model.addConstr(Gamma[i, j] <= coefficient_bound * g[i, j])
            model.addConstr(Gamma[i, j] >= -coefficient_bound * g[i, j])
            model.addConstr(1 - p + p * g[i, j] <= psi[j] - psi[i])
        else:
            model.addConstr(Gamma[i, j] == 0)
            model.addConstr(g[i, j] == 0)

    if target_status is not None:
        status = np.asarray(target_status)
        if status.shape != (environments - 1, p) or not np.isin(status, (-1, 0, 1)).all():
            raise ValueError("target_status must have shape (environments - 1, p)")
        for e, j in selectors:
            if status[e - 1, j] >= 0:
                model.addConstr(delta[e, j] == status[e - 1, j])

    def quadratic(s, j):
        return gp.quicksum(
            s[k, h] * Gamma[k, j] * Gamma[h, j]
            for k in range(p) for h in range(p)
        )

    objective = l * gp.quicksum(g.values())
    objective += weights[0] * gp.quicksum(
        -2 * log_gamma[j] + quadratic(sigma[0], j) for j in range(p)
    )
    for e, j in selectors:
        off = weights[e] * (-2 * log_gamma[j] + quadratic(sigma[e], j))
        on = weights[e] * (1 + np.log(sigma[e][j, j])) + l_delta
        model.addQConstr(theta[e, j] >= off - B[e - 1, j] * delta[e, j])
        model.addConstr(theta[e, j] >= on - B[e - 1, j] * (1 - delta[e, j]))
    model.setObjective(objective + gp.quicksum(theta.values()), GRB.MINIMIZE)
    model.Params.NonConvex, model.Params.MIPGap = 2, 1e-4
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi found no feasible solution; status {model.Status}")

    gamma = np.array([[Gamma[i, j].X for j in range(p)] for i in range(p)])
    targets = [[int(delta[e, j].X > 0.5) for j in range(p)] for e in range(1, environments)]
    return gamma, targets, model.MIPGap, model.ObjVal, model.Runtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the equation (4.4) MIP on one persisted main instance."
    )
    add_main_data_arguments(parser)
    parser.add_argument(
        "--lambda-graph",
        type=float,
        help="graph penalty; omit for the primary log(N)/N setting",
    )
    parser.add_argument("--lambda-delta", type=float)
    parser.add_argument("--screen-constant", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--metric-time-limit", type=float, default=3600)
    args = parser.parse_args()
    numeric_values = [
        args.time_limit,
        args.metric_time_limit,
        args.screen_constant,
        *(
            [] if args.lambda_graph is None else [args.lambda_graph]
        ),
        *(
            [] if args.lambda_delta is None else [args.lambda_delta]
        ),
    ]
    if (
        args.threads < 1
        or not np.isfinite(numeric_values).all()
        or args.time_limit <= 0
        or args.metric_time_limit <= 0
        or args.screen_constant <= 0
        or (args.lambda_graph is not None and args.lambda_graph < 0)
        or (args.lambda_delta is not None and args.lambda_delta < 0)
    ):
        parser.error(
            "threads must be positive and all penalties/limits must be finite "
            "with positive limits"
        )
    try:
        datasets, true_dag, true_targets, info = load_cli_instance(args)
    except MainExperimentDataError as error:
        parser.error(str(error))
    graph_penalty = (
        primary_mip_penalty(datasets)
        if args.lambda_graph is None
        else args.lambda_graph
    )
    target_penalty = (
        graph_penalty if args.lambda_delta is None else args.lambda_delta
    )
    alpha = main_screen_alpha(datasets, args.screen_constant)
    moral_graph = estimate_moral_graph(datasets[0], alpha)
    if gp is not None:
        gp.setParam("Threads", args.threads)
    result = optimization(
        datasets,
        moral_graph,
        graph_penalty,
        l_delta=target_penalty,
        time_limit=args.time_limit,
    )
    with wall_time_limit(args.metric_time_limit, "metric evaluation"):
        errors = compute_errors(result[0], result[1], true_dag, true_targets)
    print("Instance:", instance_summary(args, datasets, true_dag, true_targets, info))
    print(
        "Configuration:",
        {"screen_constant": args.screen_constant, "screen_alpha": alpha,
         "graph_penalty": graph_penalty, "target_penalty": target_penalty,
         "threads": args.threads, "fit_time_limit": args.time_limit,
         "metric_time_limit": args.metric_time_limit},
    )
    print("Gamma:\n", result[0])
    print("Targets:", result[1])
    print("MIP gap, objective, runtime:", result[2:])
    print("d_cpdag, environment target error, FDP, TDP:", errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
