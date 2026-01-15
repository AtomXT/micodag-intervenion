import numpy as np
import pandas as pd


class ComplexColliderSEM:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def generate_data(self):
        """
        Generate data according to the DAG: a -> b <- c ; d -> b
        """
        # Generate exogenous variables a and c
        a = np.random.normal(0, 2, self.n_samples)  # a ~ N(0,1)
        c = np.random.normal(0, 0.2, self.n_samples)  # c ~ N(0,1)
        d = np.random.normal(0, 0.5, self.n_samples)
        # Generate b: b = β_ab * a + β_cb * c + ε_b (collider)
        beta_ab = 0.8  # effect of a on b
        beta_cb = 0.7  # effect of c on b
        beta_db = 0.4
        epsilon_b = np.random.normal(0, 0.4, self.n_samples)
        b = beta_ab * a + beta_cb * c + beta_db * d + epsilon_b

        # Create DataFrame
        data = pd.DataFrame({
            'a': a,
            'b': b,
            'c': c,
            'd': d
        })

        true_params = {
            'beta_ab': beta_ab,
            'beta_cb': beta_cb,
            'beta_db': beta_db,
        }

        return data, true_params


def find_v_structures(adj_matrix):
    adj_matrix = (adj_matrix != 0) + 0
    np.fill_diagonal(adj_matrix, 0)
    n = len(adj_matrix)
    v_structures = set()

    # For each node y (as the common child)
    for y in range(n):
        # Find all parents of y
        parents = []
        for x in range(n):
            if adj_matrix[x][y] == 1:  # edge from x to y
                parents.append(x)

        # Check every pair of parents (x, z)
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                x = parents[i]
                z = parents[j]
                # Check no edge between x and z in either direction
                if adj_matrix[x][z] == 0 and adj_matrix[z][x] == 0:
                    # To avoid duplicates, we represent the triplet with x < z
                    if x < z:
                        v_structures.add((x, y, z))
                    else:
                        v_structures.add((z, y, x))

    return list(v_structures)


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


def Gamma_cholesky(sigma, normalized=True, alpha=None):
    L = np.linalg.cholesky(sigma)
    Gamma = np.linalg.inv(L).T
    if normalized:
        Gamma = np.round(Gamma / np.diag(Gamma), 6)
    if alpha is not None:
        Gamma[np.abs(Gamma) < alpha] = 0
    return Gamma


def ordering_triplet(sigma, alpha=0.1):
    orders = [[0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
    for order in orders:
        sigma1 = sigma[order, :][:, order]
        try:
            gamma = Gamma_cholesky(sigma1, alpha=alpha)
            if np.count_nonzero(np.triu(gamma, 1)) == 2:
                return order
        except np.linalg.LinAlgError:
            continue
    return [0,1,2]


def algorithm(X, alpha=0.1):
    n = X.shape[1]
    sigma = np.cov(X.T)
    Gamma = Gamma_cholesky(sigma, alpha=alpha)
    triplets = find_dense_triplets(Gamma)
    current_ordering = [i for i in range(n)]
    for _ in range(10):
        if triplets:
            triplet = triplets[0]
            sigma1 = sigma[triplet, :][:, triplet]
            ordering = ordering_triplet(sigma1)
            try:
                for i, idx in enumerate(triplet):
                    current_ordering[idx] = triplet[ordering[i]]
                X = X.iloc[:, current_ordering]
                sigma = np.cov(X.T)
                Gamma = Gamma_cholesky(sigma, alpha=alpha)
                triplets = find_dense_triplets(Gamma)
            except np.linalg.LinAlgError:
                triplets = triplets[1:]
    return Gamma, current_ordering


n_samples = 2000
sem = ComplexColliderSEM(n_samples)
# data, true_params = sem.generate_data()
data = pd.read_csv('~/Downloads/projects/MICODAG-CD/Data/RealWorldDatasetsTXu_largealpha/3bowling/data_3bowling_n_500_iter_1.csv', header=None)
# data = data.iloc[:, [3,1,2,0]]
final_gamma, final_ordering = algorithm(data)
print(final_gamma)
print(data.columns[final_ordering])
# print(true_params)

