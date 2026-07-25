"""Shared data-loading and graph-evaluation utilities."""

import os
import warnings

import numpy as np
import pandas as pd


def read_data(idx, iter):
    path = './Data/SyntheticData/graph' + str(idx)
    environments = [file for file in os.listdir(path) if file.startswith("environment")]
    data = []
    data_i = pd.read_csv(path + f'/observational/data_{iter}.csv', header=None)
    data.append(data_i)
    for env in environments:
        data_i = pd.read_csv(path+f'/{env}/data_{iter}.csv', header=None)
        data.append(data_i)
    p = data[0].shape[1]
    moral = pd.read_table(f'./Data/SyntheticData/Moral_{p}.txt', header=None, sep=' ')
    true_dag = pd.read_table(f'./Data/SyntheticData/DAG_{p}.txt', header=None, sep=' ')
    with open(path+'/intervention_targets.txt', 'r') as file:
        lines = file.readlines()
    interventions = [list(map(int, line.strip().split(','))) for line in lines]
    return data, moral, true_dag, interventions
def read_alpha(m, n, alpha, k):
    """
    Read data for the test on changing level of variance difference.
    :param m:
    :param n:
    :param alpha:
    :param k: number of iteration
    :return:
    """
    file_path = './Data/SyntheticDataNID_30/'
    file_name = "alpha/data_m_{}_n_{}_alpha_{}_iter_{}.csv".format(m, n, alpha, k)
    data = pd.read_csv(file_path + file_name, header=None)
    True_B = pd.read_table(file_path + "DAG_{}.txt".format(m), delimiter=" ", header=None)
    moral = pd.read_table(file_path + "Moral_DAG_{}.txt".format(m), delimiter=" ", header=None)
    mgest = pd.read_table(f'./Data/SyntheticDataNID_30/alpha/m_{m}_n_{n}_alpha_{alpha}_superstructure_glasso_iter_{k}.txt', header=None, sep=',')

    return data, True_B, moral, mgest


def ind2mat(edges, p):
    matrix = [[1 if (i, j) in set(list(map(tuple, edges))) else 0 for j in range(1, p + 1)] for i in range(1, p + 1)]
    return matrix


def tresh_cov(sigma):
    theta = np.linalg.inv(sigma)
    theta[np.abs(theta) < 0.3] = 0
    return theta


def mat2ind(mat, p):
    edges = [(i, j) for i in range(p) for j in range(p) if mat[i][j] == 1]
    return edges


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
        return graph.interventional_cpdag(target_sets, cpdag=graph.cpdag()).to_amat()[0]


def _equivalence_class(pdag, threshold=1e-6):
    """Return the directed-edge sets of all DAG extensions of a (I-)CPDAG."""
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
    """Return the max-min proportion of edges in the first DAG but not the second."""
    maximum = 0.0
    for first in first_class:
        if not first:
            distance = 0.0
        else:
            distance = min(len(first - second) / len(first) for second in second_class)
        maximum = max(maximum, distance)
    return maximum


def equivalence_class_fdp_tdp(estimated_pdag, true_pdag, threshold=1e-6):
    """Return the equivalence-class FDP and TDP from Taeb et al. (2024).

    The inputs are the estimated and true (interventional) CPDAGs. Their
    consistent DAG extensions are enumerated, and the nested max-min metrics
    in Equation (9) of the paper are evaluated on directed edge sets.
    """
    estimated_class = _equivalence_class(estimated_pdag, threshold)
    true_class = _equivalence_class(true_pdag, threshold)
    fdp = _max_min_edge_error(estimated_class, true_class)
    false_negative_proportion = _max_min_edge_error(true_class, estimated_class)
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


def find_datasets(file_path):
    lists = os.listdir(file_path)
    lists = [file for file in lists if not file.startswith('.')]
    lists = sorted(lists, key=lambda s: int(''.join(filter(str.isdigit, s))))
    return lists


def collect_results(results, datasets):
    """
    Collect results from MIP_DAG_LN_NID()
    :param results: list of results
    :param datasets: list of dataset names
    :return:
    """
    results_eq = pd.DataFrame(results['equal'], columns=['RGAP', 'd_cpdag', 'SHDs', 'FDP', 'TDP', 'Time'])
    results_eq['network'] = datasets
    results_eq = results_eq.set_index('network')
    results_ineq = pd.DataFrame(results['unequal'], columns=['RGAP', 'd_cpdag', 'SHDs', 'FDP', 'TDP', 'Time'])
    results_ineq['network'] = datasets
    results_ineq = results_ineq.set_index('network')
    return results_eq, results_ineq


def orders(lst):
    return [int(''.join(filter(str.isdigit, s))) for s in lst]


def skeleton(dag):
    """
    Given a list of arcs in the dag, return the undirected skeleton.
    This is for the computation of SHDs
    :param dag: list or arcs with 0 or 1 entries
    :return: skeleton np.array
    """
    skeleton_array = np.array(dag) + np.array(dag).T
    return skeleton_array


def compute_SHD(learned_DAG, True_DAG, SHDs=False):
    """
    Compute the stuctural Hamming distrance, which counts the number of arc differences (
    additions, deletions, or reversals)

    :param learned_DAG: list of arcs, represented as adjacency matrix
    :param True_DAG: list of arcs
    :return: shd: integer, non-negative
    """
    if type(learned_DAG) == tuple:
        learned_DAG = learned_DAG[0]
    if type(True_DAG) == tuple:
        True_DAG = True_DAG[0]
    learned_arcs = mat2ind(learned_DAG, len(learned_DAG))
    true_arcs = mat2ind(True_DAG, len(True_DAG))
    learned_skeleton = learned_arcs.copy()
    for item in learned_arcs:
        learned_skeleton.append((item[1], item[0]))
    True_skeleton = true_arcs.copy()
    for item in true_arcs:
        True_skeleton.append((item[1], item[0]))

    shd1 = len(set(learned_skeleton).difference(True_skeleton)) / 2
    shd2 = len(set((True_skeleton)).difference(learned_skeleton)) / 2
    Reversed = [(y, x) for x, y in learned_arcs]
    shd3 = len(set(true_arcs).intersection(Reversed))

    shd = shd1 + shd2 + shd3
    if SHDs:
        return shd1 + shd2
    return shd


if __name__ == '__main__':
    # print(read_B("MICP", "3bowling", "true", 0.1))
    print('Running utils.')
