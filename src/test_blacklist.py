import numpy as np
from typing import List, Tuple, Set
from cd_spacer import CD_order
from itertools import permutations, islice
import random
from cd_spacer import *

import pandas as pd
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

def Gamma_cholesky(sigma, normalized=True, alpha=None):
    L = np.linalg.cholesky(sigma)
    Gamma = np.linalg.inv(L).T
    if normalized:
        Gamma = np.round(Gamma / np.diag(Gamma), 6)
    if alpha is not None:
        Gamma[np.abs(Gamma) < alpha] = 0
    return Gamma


def find_dense_triplets(adj_matrix):
    adj_matrix = (adj_matrix != 0) + 0
    np.fill_diagonal(adj_matrix, 0)
    n = len(adj_matrix)
    triplets = set()
    for a in range(n):
        for b in range(n):
            if adj_matrix[a][b] == 1:  # a->b
                for c in range(n):
                    if adj_matrix[b][c] == 1 and adj_matrix[a][c] == 1:  # b->c and a->c
                        # Represent the triplet as sorted tuple to avoid duplicates
                        nodes = tuple(sorted([a, b, c]))
                        triplets.add(nodes)
    return list(triplets)



def get_n_permutations(lst, n):
    """
    Efficient random permutations without generating all possibilities
    Good for large lists
    """
    result = []
    for _ in range(n):
        # Create a random permutation by shuffling
        shuffled = lst[:]  # Copy the list
        random.shuffle(shuffled)
        result.append(tuple(shuffled))
    return result



def main_algorithm(X: np.ndarray) -> np.ndarray:
    """
    Main algorithm that reorders variables based on triplet analysis.

    Args:
        X: Input dataset with shape (n, p)

    Returns:
        Reordered dataset
    """
    # Initialize
    current_X = X.copy()
    p = X.shape[1]  # number of variables
    A = np.ones((p, p))
    np.fill_diagonal(A, 0)
    orderings = get_n_permutations(list(range(p)), 100)


    # Continue until convergence
    for i, ordering in enumerate(orderings):
        print(f"The {i}th permutation is {ordering}.")
        X1 = X[:, ordering]
        back = np.argsort(ordering)
        Gamma = Gamma_cholesky(np.cov(X1.T), alpha=0.05)
        for i in range(p):
            for j in range(i+1, p):
                if Gamma[i][j] == 0:
                    A[ordering[i]][ordering[j]] = 0
    return A


def reorder_dataset(X: np.ndarray, current_ordering: List[int],
                    best_order: Tuple[int, int, int], triplet: Tuple[int, int, int]) -> Tuple[np.ndarray, List[int]]:
    """
    Reorder the dataset based on the best order found and update the ordering mapping.

    Args:
        X: Current dataset
        current_ordering: Current mapping from position to original index
        best_order: Optimal ordering for the triplet (3 original indices)
        triplet: Original triplet indices that were tested

    Returns:
        Tuple of (reordered dataset, updated ordering mapping)
    """
    p = X.shape[1]

    # Convert best_order (original indices) to current positions
    best_order_positions = [current_ordering.index(idx) for idx in triplet]

    # Create new ordering
    new_ordering = np.array(current_ordering.copy())
    new_ordering[best_order_positions] = best_order

    # Create the new dataset
    new_X = X[:, new_ordering]

    return new_X, list(new_ordering)


def is_sparse(Gamma: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Determine if a Gamma matrix is sparse.

    Args:
        Gamma: The matrix to check
        threshold: Fraction of non-zero elements to consider as sparse

    Returns:
        True if matrix is sparse, False otherwise
    """
    # Example implementation: check if most elements are near zero
    # You might want to adjust this based on your definition of "dense"
    non_zero_count = np.count_nonzero(np.triu(Gamma, 1))

    return non_zero_count == 2


def create_complete_ordering(X: np.ndarray, best_order: Tuple[int, int, int],
                             triplet: Tuple[int, int, int]) -> List[int]:
    """
    Create a complete reordering of all variables based on the best order found.

    Args:
        X: Current dataset
        best_order: Optimal ordering for the triplet (3 indices)
        triplet: Original triplet indices

    Returns:
        Complete reordering of all variable indices
    """
    p = X.shape[1]
    current_order = list(range(p))

    # Remove the triplet indices from current order
    remaining_indices = [i for i in current_order if i not in triplet]

    # Insert the best ordered triplet at the appropriate position
    # This is a simple implementation - you might want to customize this
    # based on how you want to position the reordered triplet

    # For example, insert at the beginning:
    new_order = list(best_order) + remaining_indices

    return new_order


def read_data(network, n=500, iter=1):
    folder_path = os.path.join(current_dir, "../Data/RealWorldDatasets/")
    data_path = folder_path + f"{network}/data_{network}_n_{n}_iter_{iter}.csv"
    file_path = folder_path + f"{network}"
    graph_name = [i for i in os.listdir(
        file_path) if os.path.isfile(os.path.join(file_path, i)) and 'Sparse_Original_edges' in i][0]
    graph_path = folder_path + network + f"/{graph_name}"
    moral_graph_name = [i for i in os.listdir(
        file_path) if os.path.isfile(os.path.join(file_path, i)) and 'Sparse_Moral_edges' in i][0]
    moral_path = folder_path + network + f"/superstructure_glasso_iter_{iter}.txt"
    true_moral_path = folder_path + network + f"/{moral_graph_name}"
    data, graph = pd.read_csv(data_path, header=None), pd.read_table(graph_path, delimiter=',', dtype=int, header=None)
    moral = pd.read_table(moral_path, delimiter=',', dtype=int, header=None)
    true_moral = pd.read_table(true_moral_path, delimiter=',', dtype=int, header=None)
    graph_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    true_moral_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    for i in range(len(graph)):
        graph_[graph.iloc[i, 0]-1][graph.iloc[i, 1]-1] = 1
    for i in range(len(true_moral)):
        true_moral_[true_moral.iloc[i, 0] - 1][true_moral.iloc[i, 1] - 1] = 1
    graph_, true_moral_ = np.array(graph_), np.array(true_moral_)
    return data, graph_, moral.values, true_moral_


# Example usage
if __name__ == "__main__":
    # Create sample data
    data, true_dag, moral_lasso, true_moral = read_data("10factors", n=500, iter=1)
    N, P = data.shape
    true_moral = true_moral + true_moral.T
    # order = [5,4,3,2,1,0]
    # data = data.iloc[:, order]

    # Run the algorithm
    graph = main_algorithm(data.values)
    graph = graph + graph.T
    graph = (graph != 0) + 0

    true_dag_ = cd.DAG.from_amat(true_dag)
    true_cpdag = true_dag_.cpdag().to_amat()
    # estimated_dag = cd.DAG.from_amat(graph)
    # estimated_cpdag = estimated_dag.cpdag().to_amat()
    estimated_cpdag = graph
    SHD_cpdag = np.sum(np.abs(estimated_cpdag - true_cpdag[0]))
    est_ = graph
    skeleton_estimated, skeleton_true = skeleton(est_), skeleton(true_dag)
    SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
    TPR = np.sum(np.logical_and(est_, true_dag)) / np.sum(true_dag)
    FPR = (np.sum(est_) - np.sum(np.logical_and(est_, true_dag))) / (P * P - np.sum(true_dag))
    print(
        f"TPR: {TPR}; FPR:{FPR}; shd_cpdag: {SHD_cpdag}; SHDs: {SHDs}; ")
    # print(true_dag)

    # print(np.array(final_ordering)[np.argsort(order)])
    # print(graph)