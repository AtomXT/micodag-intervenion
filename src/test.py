import numpy as np
from typing import List, Tuple, Set
from cd_spacer import CD_order

import pandas as pd
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


    # Track the current ordering relative to original data
    # current_ordering[i] = original index of variable at position i
    current_ordering = list(range(p))

    # Continue until convergence
    for _ in range(1000):
        # Step 1: Get Gamma matrix using Gamma_cholesky
        Gamma = Gamma_cholesky(np.cov(current_X.T), alpha=0.1)

        # Step 2: Get list of triplets using find_dense_triplets
        triplets = find_dense_triplets(Gamma)

        # If no triplets found, break
        if not triplets:
            break

        # Track if any triplet was successfully processed
        any_triplet_processed = False

        # Process each triplet
        for triplet in triplets:
            current_triplet = [current_ordering[i] for i in triplet]

            # Try all 6 possible orderings of the triplet
            all_orders = [
                (current_triplet[0], current_triplet[1], current_triplet[2]),
                (current_triplet[0], current_triplet[2], current_triplet[1]),
                (current_triplet[1], current_triplet[0], current_triplet[2]),
                (current_triplet[1], current_triplet[2], current_triplet[0]),
                (current_triplet[2], current_triplet[0], current_triplet[1]),
                (current_triplet[2], current_triplet[1], current_triplet[0])
            ]

            found_sparse_order = False
            best_order = None

            # Test each ordering
            for order in all_orders:
                # Get the column indices in current ordering for this specific order
                order_indices = [current_ordering.index(idx) for idx in order]

                # Get sub-dataset with only these three columns in current order
                X_sub = current_X[:, order_indices]

                # Get Gamma for sub-dataset
                Gamma_sub = Gamma_cholesky(np.cov(X_sub.T), alpha=0.1)

                # Check if Gamma_sub is sparse (implementation depends on your definition)
                if is_sparse(Gamma_sub):
                    found_sparse_order = True
                    best_order = order
                    break  # Found a sparse ordering, no need to check others

            # If we found a sparse ordering for this triplet
            if found_sparse_order:
                any_triplet_processed = True

                # Reorder the dataset and update the ordering mapping
                current_X, current_ordering = reorder_dataset(current_X, current_ordering,
                                                              best_order, current_triplet)

                # Break out of triplet loop to restart with new ordering
                break

        # If no triplet was processed in this iteration, we're done
        if not any_triplet_processed:
            break

    # Create the reverse mapping: final_ordering[i] = original index at position i
    final_ordering = current_ordering

    # Also create mapping: original_index -> final_position
    mapping_to_final = {original_idx: final_pos for final_pos, original_idx in enumerate(final_ordering)}

    return current_X, final_ordering, mapping_to_final


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


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = pd.read_csv(
        '~/Downloads/projects/MICODAG-CD/Data/RealWorldDatasetsTXu_largealpha/1dsep/data_1dsep_n_500_iter_1.csv',
        header=None)
    order = [5,4,3,2,1,0]
    data = data.iloc[:, order]

    # Run the algorithm
    reordered_X, final_ordering, _ = main_algorithm(data.values)

    data = data.iloc[:, np.array(final_ordering)[np.argsort(order)]]
    n, p = data.shape
    moral = np.ones((p, p))
    moral = np.triu(moral, k=1)
    est, _ = CD_order(data, moral, lam=np.sqrt(20*np.log(p)/n))

    print(np.array(final_ordering)[np.argsort(order)])
    print(est)