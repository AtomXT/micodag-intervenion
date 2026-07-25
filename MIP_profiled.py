#!/usr/bin/env python3
"""Fully profiled parent-set MIP for unknown intervention targets."""

from __future__ import annotations

import argparse
from itertools import combinations
from math import comb
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from MIP import _edges, _load, _moments, estimate_moral_graph
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
):
    """Return Gamma, targets, MIP gap, objective, and runtime.

    If ``l_delta`` is omitted, environment ``e`` uses the intervention
    penalty ``l * weights[e]``. A supplied scalar applies the same penalty
    to every interventional environment; a vector supplies one penalty per
    interventional environment.
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
    return gamma, targets.tolist(), model.MIPGap, model.ObjVal, model.Runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=int, default=2)
    parser.add_argument("--iteration", type=int, default=6)
    parser.add_argument("--lambda-graph", type=float, default=0.1)
    parser.add_argument(
        "--lambda-delta",
        type=float,
        help="common target penalty; omit to use lambda_graph times each environment weight",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-parents", type=int)
    parser.add_argument("--time-limit", type=float, default=60)
    args = parser.parse_args()

    data, _, true_dag, true_targets = _load(
        Path("data/SyntheticData"), args.graph, args.iteration
    )
    moral = estimate_moral_graph(data[0], args.alpha)
    result = optimization(
        data, moral, args.lambda_graph, l_delta=args.lambda_delta,
        max_parents=args.max_parents, time_limit=args.time_limit,
    )
    errors = compute_errors(result[0], result[1], true_dag, true_targets)
    print("Gamma:\n", result[0])
    print("Targets:", result[1])
    print("MIP gap, objective, runtime:", result[2:])
    print("d_cpdag, environment target error, FDP, TDP:", errors)
