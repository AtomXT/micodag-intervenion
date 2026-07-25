#!/usr/bin/env python3
"""Equation (4.4): DAG learning with unknown intervention targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.covariance import GraphicalLasso
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


def _load(root, graph, iteration):
    folder = root / f"graph{graph}"
    paths = sorted(folder.glob("environment*"), key=lambda x: int(x.name[11:]))
    data = [np.loadtxt(folder / "observational" / f"data_{iteration}.csv", delimiter=",")]
    data += [np.loadtxt(x / f"data_{iteration}.csv", delimiter=",") for x in paths]
    p = data[0].shape[1]
    moral = np.loadtxt(root / f"Moral_{p}.txt", ndmin=2)
    true_dag = np.zeros((p, p), dtype=int)
    true_dag[tuple((np.loadtxt(root / f"DAG_{p}.txt", dtype=int, ndmin=2) - 1).T)] = 1
    true_targets = np.zeros((len(paths), p), dtype=int)
    for e, line in enumerate((folder / "intervention_targets.txt").read_text().splitlines()):
        true_targets[e, np.fromstring(line, dtype=int, sep=",") - 1] = 1
    return data, moral, true_dag, true_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--lambda-graph", type=float, default=0.05)
    parser.add_argument("--lambda-delta", type=float)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--time-limit", type=float, default=60)
    args = parser.parse_args()
    datasets, _, true_dag, true_targets = _load(Path("data/SyntheticData"), args.graph, args.iteration)
    moral_graph = estimate_moral_graph(datasets[0], args.alpha)
    result = optimization(
        datasets, moral_graph, args.lambda_graph,
        l_delta=args.lambda_delta, time_limit=args.time_limit,
    )
    errors = compute_errors(result[0], result[1], true_dag, true_targets)
    print("Gamma:\n", result[0])
    print("Targets:", result[1])
    print("MIP gap, objective, runtime:", result[2:])
    print("d_cpdag, environment target error, TPR, FPR:", errors)
