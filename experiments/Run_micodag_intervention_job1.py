import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import nlfunc
from gurobipy import GRB
import os
import causaldag as cd
from utils import ind2mat, skeleton, performance

current_dir = os.path.dirname(os.path.abspath(__file__))

def read_data(idx, iter):
    path = f'{current_dir}/../data/SyntheticData/graph' + str(idx)
    environments = [file for file in os.listdir(path) if file.startswith("environment")]
    data = []
    data_i = pd.read_csv(path + f'/observational/data_{iter}.csv', header=None)
    data.append(data_i)
    for env in environments:
        data_i = pd.read_csv(path+f'/{env}/data_{iter}.csv', header=None)
        data.append(data_i)
    p = data[0].shape[1]
    moral = pd.read_table(f'{current_dir}/../data/SyntheticData/Moral_{p}.txt', header=None, sep=' ')
    true_dag = pd.read_table(f'{current_dir}/../data/SyntheticData/DAG_{p}.txt', header=None, sep=' ')
    with open(path+'/intervention_targets.txt', 'r') as file:
        lines = file.readlines()
    interventions = [list(map(int, line.strip().split(','))) for line in lines]
    return data, moral, true_dag, interventions


def solve(Xs, lam, mu, moral, weights=None):
    # data preparation
    n_env = len(Xs)  # number of environment
    m = Xs[0].shape[1]  # dimension
    n_data = [Xs[e].shape[0] for e in range(n_env)]  # number of samples in each environment
    Sigmas = [np.cov(Xs[e].T) for e in range(n_env)]  # sample covariances
    if not weights:
        weights = np.array([n/np.sum(n_data) for n in n_data])
    M = 100  # big M

    model = gp.Model()
    Gammas = [model.addMVar((m, m), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) for _ in range(n_env)]
    gs = [model.addMVar((m, m), vtype=GRB.BINARY) for _ in range(n_env)]
    psis = [model.addMVar((m, 1), lb=1, ub=m, vtype=GRB.CONTINUOUS, name='psi') for _ in range(n_env)]
    log_term = model.addMVar(n_env, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    t = [model.addMVar(m, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) for _ in range(n_env)]

    for e in range(n_env):
        model.addConstrs(Gammas[e][i, i] <= M for i in range(m))
        model.addConstrs(Gammas[e][i, i] >= 0 for i in range(m))
        for j in range(m):
            for k in range(m):
                if j != k:
                    if moral[j, k] == 1:
                        model.addConstr(Gammas[e][j, k] <= M * gs[e][j, k])  # encoding moral graph
                        model.addConstr(Gammas[e][j, k] >= -M * gs[e][j, k])
                    else:
                        model.addConstr(Gammas[e][j, k] == 0)
                        model.addConstr(gs[e][j, k] == 0)
                    model.addConstr(1-m+m*gs[e][j, k] <= psis[e][k] - psis[e][j])  # DAG constraint

    for e in range(n_env):
        model.addConstr(log_term[e] == gp.quicksum(-2*nlfunc.log(Gammas[e][i, i]) for i in range(m)))

    traces = [gp.QuadExpr() for _ in range(n_env)]
    for e in range(n_env):
        # for k in range(p):
        #     for i in range(p):
        #         for j in range(p):
        #             traces[e] += Gammas[e][j, k] * Gammas[e][j, i] * Sigmas[e][i, k]
        temp = Gammas[e]@Gammas[e].T@Sigmas[e]
        traces[e] = gp.quicksum(temp[i, i] for i in range(m))

    penalty1 = lam * gp.quicksum(gs[0][i, j] if i != j else 0 for i in range(m) for j in range(m))
    for e in range(1, n_env):
        for j in range(m):
            model.addConstr(gp.quicksum((Gammas[e][i, j] - Gammas[0][i, j]) ** 2 for i in range(m)) <= t[e][j] * t[e][j])


    penalty2 = mu * gp.quicksum(t[e][j] for e in range(1, n_env) for j in range(m))
    # penalty2 = 0
    penalty = penalty1 + penalty2

    # model.addConstr(np.sum(weights[e] * traces[e] for e in range(n_env)) >= m - 0.5)  # necessary condition
    # model.addConstr(np.sum(weights[e] * traces[e] for e in range(n_env)) <= m + 0.5)

    model.setObjective(gp.quicksum(weights[e]*(log_term[e]+traces[e]) for e in range(n_env)) + penalty, GRB.MINIMIZE)
    model.params.TimeLimit = 50*m
    model.setParam('MIPGapAbs', lam*m)
    model.params.Threads = 16
    model.optimize()

    return [Gammas[e].X for e in range(n_env)], model.Runtime, model.MIPGap


# graph = 2
# iter = 1
lam = 0.05
mu = 0.1
results = []
for graph in [1, 2]:
    for iter in range(1, 11):
        datas, moral_graph, true_graph, interventions = read_data(graph, iter)
        n, p = datas[0].shape
        true_moral_ = np.array(ind2mat(moral_graph.values, p))
        true_moral_ += true_moral_.transpose()
        true_graph_ = np.array(ind2mat(true_graph.values, p))


        gammas, run_time, rgap = solve(datas, lam, mu, true_moral_)

        true_dag = cd.DAG.from_amat(true_graph_)
        true_cpdag = true_dag.cpdag().to_amat()
        B_arcs = [[0 if np.abs(gammas[0][i][j]) <= 1e-6 or i == j else 1 for j in range(p)] for i in range(p)]
        estimated_dag = cd.DAG.from_amat(np.array(B_arcs))
        estimated_cpdag = estimated_dag.cpdag().to_amat()
        SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))

        # SHD = compute_SHD(B_arcs, True_B_mat)
        skeleton_estimated, skeleton_true = skeleton(B_arcs), skeleton(true_graph_)
        TPR, FPR = performance(skeleton_estimated, skeleton_true)
        print(np.count_nonzero(B_arcs), np.count_nonzero(true_graph_))
        print(TPR, FPR, SHD_cpdag)
        results.append([graph, iter, lam, mu, rgap, TPR, FPR, SHD_cpdag, run_time])
        results_df = pd.DataFrame(results,
                                  columns=['graph', 'iter', 'lam', 'mu', 'rgap', 'TPR', 'FPR',
                                           'd_cpdag', 'time'])
        print(results_df)
        results_df.to_csv(f"{current_dir}/../experiment_results/micodag_intervention_job1.csv", index=False)

        # np.sum(datas[e].shape[0]/600*(-2*np.sum(np.log(np.diag(gammas[e]))) + np.trace(gammas[e]@gammas[e].T@np.cov(datas[e].T))) for e in range(6))
        # np.sum(datas[e].shape[0]/600*(np.trace(gammas[e]@gammas[e].T@np.cov(datas[e].T))) for e in range(6))
        # np.trace(gammas[0]@gammas[0].T@np.cov(datas[0].T))
        # np.trace(gammas[1]@gammas[1].T@np.cov(datas[1].T))