

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


import numpy as np
import pandas as pd
import os

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


def performance(A, Theta):
    """
    parameters:

    A: The estimated matrix.
    Theta: The ground truth matrix.

    """
    A, Theta = np.array(A), np.array(Theta)
    support_Theta = Theta != 0
    support_A = A != 0
    P = np.count_nonzero(support_Theta)
    p = Theta.shape[0]
    N = p*(p-1) - P
    TP = np.count_nonzero(np.multiply(support_Theta, support_A))
    TPR = TP / P
    FP = np.count_nonzero(support_A) - TP
    FPR = FP / N
    return TPR, FPR


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
    results_eq = pd.DataFrame(results['equal'], columns=['RGAP', 'd_cpdag', 'SHDs', 'TPR', 'FPR', 'Time'])
    results_eq['network'] = datasets
    results_eq = results_eq.set_index('network')
    results_ineq = pd.DataFrame(results['unequal'], columns=['RGAP', 'd_cpdag', 'SHDs', 'TPR', 'FPR', 'Time'])
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



