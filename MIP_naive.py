#!/usr/bin/env python3
"""Naive joint-environment formulation from Equation (4.2)."""

import argparse
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB, nlfunc

from MIP import _edges, _load, _moments, compute_errors, estimate_moral_graph


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
    time_limit=None,
):
    """Return baseline Gamma, targets, MIP gap, objective, and runtime."""
    l_delta = l if l_delta is None else l_delta
    data, sigma, weights = _moments(data)
    environments, p = len(data), data[0].shape[1]
    candidates = _edges(moral, p)
    arcs = [(i, j) for i in range(p) for j in range(p) if i != j]
    selectors = [(e, j) for e in range(1, environments) for j in range(p)]

    model = gp.Model("naive_interventional_dag_mip")
    Gamma = model.addVars(
        environments, p, p, lb=-coefficient_bound, ub=coefficient_bound, name="Gamma"
    )
    log_gamma = model.addVars(
        environments, p, lb=np.log(gamma_lower), ub=np.log(gamma_upper), name="log_Gamma"
    )
    g = model.addVars(arcs, vtype=GRB.BINARY, name="g")
    psi = model.addVars(p, lb=1, ub=p, name="psi")
    delta = model.addVars(selectors, vtype=GRB.BINARY, name="delta")

    for e in range(environments):
        for j in range(p):
            Gamma[e, j, j].LB, Gamma[e, j, j].UB = gamma_lower, gamma_upper
            model.addGenConstrNL(log_gamma[e, j], nlfunc.log(Gamma[e, j, j]))
    for i, j in arcs:
        if (i, j) in candidates:
            model.addConstr(Gamma[0, i, j] <= coefficient_bound * g[i, j])
            model.addConstr(Gamma[0, i, j] >= -coefficient_bound * g[i, j])
            model.addConstr(1 - p + p * g[i, j] <= psi[j] - psi[i])
        else:
            model.addConstr(Gamma[0, i, j] == 0)
            model.addConstr(g[i, j] == 0)

    for e, j in selectors:
        for i in range(p):
            difference = Gamma[e, i, j] - Gamma[0, i, j]
            bound = gamma_upper - gamma_lower if i == j else coefficient_bound
            model.addConstr(difference <= bound * delta[e, j])
            model.addConstr(difference >= -bound * delta[e, j])
            if i != j:
                model.addConstr(Gamma[e, i, j] <= coefficient_bound * (1 - delta[e, j]))
                model.addConstr(Gamma[e, i, j] >= -coefficient_bound * (1 - delta[e, j]))

    if target_status is not None:
        status = np.asarray(target_status)
        if status.shape != (environments - 1, p) or not np.isin(status, (-1, 0, 1)).all():
            raise ValueError("target_status must have shape (environments - 1, p)")
        for e, j in selectors:
            if status[e - 1, j] >= 0:
                model.addConstr(delta[e, j] == status[e - 1, j])

    def quadratic(e, j):
        return gp.quicksum(
            sigma[e][k, h] * Gamma[e, k, j] * Gamma[e, h, j]
            for k in range(p) for h in range(p)
        )

    objective = l * gp.quicksum(g.values()) + l_delta * gp.quicksum(delta.values())
    objective += gp.quicksum(
        weights[e] * (-2 * log_gamma[e, j] + quadratic(e, j))
        for e in range(environments) for j in range(p)
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.Params.NonConvex, model.Params.MIPGap = 2, 1e-4
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi found no feasible solution; status {model.Status}")

    gamma = np.array([[Gamma[0, i, j].X for j in range(p)] for i in range(p)])
    targets = [[int(delta[e, j].X > 0.5) for j in range(p)] for e in range(1, environments)]
    return gamma, targets, model.MIPGap, model.ObjVal, model.Runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=6)
    parser.add_argument("--lambda-graph", type=float, default=0.05)
    parser.add_argument("--lambda-delta", type=float)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--time-limit", type=float, default=60)
    args = parser.parse_args()

    data, _, true_dag, true_targets = _load(Path("data/SyntheticData"), args.graph, args.iteration)
    moral = estimate_moral_graph(data[0], args.alpha)
    result = optimization(
        data, moral, args.lambda_graph, l_delta=args.lambda_delta, time_limit=args.time_limit
    )
    errors = compute_errors(result[0], result[1], true_dag, true_targets)
    print("Gamma:\n", result[0])
    print("Targets:", result[1])
    print("MIP gap, objective, runtime:", result[2:])
    print("d_cpdag, environment target error, TPR, FPR:", errors)
