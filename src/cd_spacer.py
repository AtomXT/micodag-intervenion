from collections import deque
import numpy as np
import pandas as pd
import causaldag as cd
from collections import defaultdict
import matplotlib.pyplot as plt
import time


def cycle(G, i, j):
    """
    Check whether a DAG G remains acyclic if an edge i->j is added.
    Return True if it is no longer a DAG.

    Examples
    --------
    Consider a DAG defined as:
    dag = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    print(cycle(dag, 3, 2))
    """
    P = len(G)
    C = [0] * P
    C[i] = 1
    Q = deque([i])
    while Q:
        u = Q.popleft()
        parent_u = [ii for ii in range(P) if G[ii, u] != 0]
        for v in parent_u:
            if v == j:
                return True
            else:
                if C[v] == 0:
                    C[v] = 1
                    Q.append(v)
    return False


def g(Gamma, sigma_hat, u, v, lam):
    ans = sum([Gamma[u, v]*Gamma[i, v]*sigma_hat[i, u] for i in range(len(Gamma))])
    ans += Gamma[v, v]*Gamma[u, v]*sigma_hat[u, v]
    ans += sum([Gamma[k, v] * Gamma[u, v] * sigma_hat[u, k] for k in range(len(Gamma)) if k != u and k != v])
    if Gamma[u, v] != 0:
        ans += lam*lam
    return ans


def objective(Gamma, sigma_hat, lam):
    P = len(Gamma)
    obj = sum([-2*np.log(Gamma[i, i]) for i in range(P)])
    obj += np.trace(Gamma@Gamma.T@sigma_hat)
    obj += lam*lam*(np.count_nonzero(Gamma)-P)
    return obj


def gamma_hat(Gamma, sigma_hat, u, v, lam):
    # numerator = sum([Gamma[i, v]*(sigma_hat[i, u]+sigma_hat[u, i]) for i in range(len(Gamma)) if i != u])
    numerator = sum([2 * Gamma[i, v] * sigma_hat[i, u] for i in range(len(Gamma)) if i != u])
    # return -numerator/sigma_hat[u, u]/2 if g(gamma, sigma_hat, u, v, lam) < 0 else 0
    return -numerator/sigma_hat[u, u]/2 if lam*lam <= numerator*numerator/(4*sigma_hat[u,u]) else 0


def gamma_hat_diag(Gamma, sigma_hat, u):
    # star = sum([Gamma[j, u]*(sigma_hat[j, u]+sigma_hat[u, j]) for j in range(len(Gamma)) if j != u])
    star = sum([2 * Gamma[j, u] * sigma_hat[j, u]for j in range(len(Gamma)) if j != u])
    return (-star + np.sqrt(star*star + 16*sigma_hat[u, u]))/(4*sigma_hat[u, u])


def CD(X, moral, lam=0.01, MAX_cycles=100, tol=1e-2):
    """

    X: N by P data matrix
    """
    N, P = X.shape
    sigma = np.cov(X.T)
    # sigma = (X.T@X/N).values
    Gamma = np.eye(P)
    objs = []
    min_obj = float('inf')
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        if t % 10 == 0:
            print(f"cycle: {t}")
        for u in range(P):
            Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            for v in range(P):
                if u != v and moral[u, v] == 1:
                    temp_gamma = Gamma.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    if cycle_uv:
                        Gamma[u, v] = 0
                    else:
                        Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t < tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)

    # print(objs)
    # print(objsi)
    return opt_Gamma, min_obj


def EqVarDAG_TD_internal(X):
    n, p = X.shape
    done = [p]  # Initialize with an invalid index (p is out of bounds)
    S = np.cov(X, rowvar=False)  # Compute covariance matrix
    Sinv = np.linalg.inv(S)  # Compute inverse of the covariance matrix

    for i in range(p):
        # Exclude already selected variables
        remaining = [j for j in range(p) if j not in done]
        # Submatrix of Sinv for remaining variables
        Sinv_sub = Sinv[np.ix_(remaining, remaining)]
        # Compute diagonal of the inverse of the submatrix
        diag_inv = np.diag(np.linalg.inv(Sinv_sub))
        # Find the variable with the minimum diagonal value
        v = remaining[np.argmin(diag_inv)]
        done.append(v)

    # Remove the initial invalid index (p) and return the topological order
    return {'TO': done[1:], 'support': None}


def CD_order(X, moral, ordering=None, lam=0.01, MAX_cycles=100, tol=1e-6, start=None, cholesky=False):
    """

    X: N by P data matrix
    """
    if ordering is None:
        TO = np.array(EqVarDAG_TD_internal(X)['TO'])
        # TO = np.array([i for i in range(X.shape[1])])
    else:
        TO = np.argsort(ordering)  # consistent for 0-based or 1-based indexing
    original_order = np.argsort(TO)
    N, P = X.shape

    # Reorder the adjacency matrix
    moral = moral[TO, :][:, TO]
    X = X[list(X.columns[TO])]
    sigma = np.cov(X.T)
    if start is not None:
        start = start[TO, :][:, TO]
    if cholesky:
        start = np.linalg.inv(np.linalg.cholesky(np.cov(X.T)).T)
        diag_Sigma = np.sqrt(np.diag(sigma))  # shape (p,)

        # Broadcast threshold matrix: each row u is lambda / Sigma[u,u]
        thresholds = 2*lam / diag_Sigma[:, np.newaxis]  # shape (p, 1)

        # Apply thresholding
        start = start * (np.abs(start) >= thresholds)
        # start[np.abs(start) < 0.3] = 0



    # sigma = (X.T@X/N).values
    Gamma = np.eye(P) if start is None else start.copy()
    objs = []
    min_obj = float('inf')
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        if t % 10 == 0:
            print(f"cycle: {t}")
        for u in range(P):
            Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            for v in range(P):
                if u != v:
                    if moral[u, v] == 1:
                        temp_gamma = Gamma.copy()
                        temp_gamma -= np.diag(np.diag(temp_gamma))
                        cycle_uv = cycle(temp_gamma, u, v)
                        if cycle_uv:
                            Gamma[u, v] = 0
                        else:
                            Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                    else:
                        Gamma[u, v] = 0
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t < tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)
        # print(opt_Gamma)

    # print(objs)
    # print(objsi)

    # back to the original order
    opt_Gamma = opt_Gamma[original_order, :][:, original_order]

    return opt_Gamma, min_obj


from causaldag import rand, partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester, gsp
def topological_sort(adj_matrix):
    n = len(adj_matrix)
    in_degree = [0] * n

    # Step 1: Compute in-degrees
    for j in range(n):
        for i in range(n):
            if adj_matrix[i][j]:
                in_degree[j] += 1

    # Step 2: Initialize queue with nodes of in-degree 0
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in range(n):
            if adj_matrix[u][v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    if len(topo_order) != n:
        raise ValueError("The graph is not a DAG (contains a cycle).")

    return np.array(topo_order)
def CD_GSP_order(X, moral, lam=0.01, MAX_cycles=100, tol=1e-6, start=None, cholesky=False):
    """

    X: N by P data matrix
    """
    N, P = X.shape

    indices = np.where(moral != 0)
    possible_edges_est = tuple(zip(indices[0], indices[1]))
    fixed_gaps_est = set()
    for i in range(P):
        for j in range(P):
            if (i, j) not in possible_edges_est:
                fixed_gaps_est.add((i, j))

    suffstat = partial_correlation_suffstat(X)
    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)
    est_dag = gsp(set(range(P)), ci_tester, fixed_gaps=fixed_gaps_est)
    B_arcs = est_dag.to_amat()[0]
    TO = topological_sort(B_arcs)

    # TO = np.array(EqVarDAG_TD_internal(X)['TO'])
    # TO = np.array([i for i in range(X.shape[1])])
    original_order = np.argsort(TO)


    # Reorder the adjacency matrix
    moral = moral[TO, :][:, TO]
    X = X[list(X.columns[TO])]
    sigma = np.cov(X.T)
    if start is not None:
        start = start[TO, :][:, TO]
    if cholesky:
        start = np.linalg.inv(np.linalg.cholesky(np.cov(X.T)).T)
        diag_Sigma = np.sqrt(np.diag(sigma))  # shape (p,)

        # Broadcast threshold matrix: each row u is lambda / Sigma[u,u]
        thresholds = 2*lam / diag_Sigma[:, np.newaxis]  # shape (p, 1)

        # Apply thresholding
        # start = start * (np.abs(start) >= thresholds)
        # start[np.abs(start) < 0.3] = 0



    # sigma = (X.T@X/N).values
    Gamma = np.eye(P) if start is None else start.copy()
    objs = []
    min_obj = float('inf')
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        if t % 10 == 0:
            print(f"cycle: {t}")
        for u in range(P):
            Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            for v in range(P):
                if u != v:
                    if moral[u, v] == 1:
                        temp_gamma = Gamma.copy()
                        temp_gamma -= np.diag(np.diag(temp_gamma))
                        cycle_uv = cycle(temp_gamma, u, v)
                        if cycle_uv:
                            Gamma[u, v] = 0
                        else:
                            Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                    else:
                        Gamma[u, v] = 0
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t < tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)
        # print(opt_Gamma)

    # print(objs)
    # print(objsi)

    # back to the original order
    opt_Gamma = opt_Gamma[original_order, :][:, original_order]

    return opt_Gamma, min_obj


# def CD_cyclic(X, lam=0.01, MAX_cycles=10):
#     """
#
#     X: N by P data matrix
#     """
#     N, P = X.shape
#     sigma = np.cov(X.T)
#     Gamma = np.eye(P)
#     objs = []
#     min_obj = float('inf')
#     opt_Gamma = None
#     for t in range(MAX_cycles):
#         if t % 10 == 0:
#             print(f"cycle: {t}")
#         for u in range(P):
#             Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u, 10)
#             for v in range(u+1, P):
#                 Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
#                 Gamma[v, u] = gamma_hat(Gamma, sigma, v, u, lam)
#         obj_t = objective(Gamma, sigma, lam)
#         # if obj_t in objs[-10:] and t > 200:
#         #     print(f"stop at the {t}-th iteration.")
#         #     break
#         if obj_t < min_obj:
#             min_obj = obj_t
#             opt_Gamma = Gamma
#         objs.append(obj_t)
#     print(objs)
#     print(f"Best objective: {min_obj}")
#     return opt_Gamma


def read_data(network, n=500, iter=1):
    folder_path = "./Data/RealWorldDatasets/"
    data_path = folder_path + f"{network}/data_{network}_n_{n}_iter_{iter}.csv"
    graph_path = folder_path + network + "/Sparse_Original_edges.txt"
    moral_path = folder_path + network + f"/superstructure_glasso_iter_{iter}.txt"
    true_moral_path = folder_path + network + "/Sparse_Moral_edges.txt"
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
    return data, graph_, moral, true_moral_


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


def mat2ind(mat, p):
    edges = [(i, j) for i in range(p) for j in range(p) if mat[i][j] == 1]
    return edges


if __name__ == '__main__':
    data, true_dag, moral_lasso, true_moral = read_data("1asia", n=500, iter=1)
    N, P = data.shape
    true_moral = true_moral + true_moral.T
    # ones = np.ones((P, P))
    # ones = ones - np.diag(np.diag(ones))
    start_i = time.time()
    est, min_obj = CD(data, true_moral, MAX_cycles=400, lam=np.sqrt(12*np.log(P)/N)) # np.sqrt(12*np.log(P)/N) np.sqrt(np.log(N)/np.sqrt(N))/2
    end_i = time.time()
    est_ = np.array([[1 if i != j and est[i, j] != 0 else 0 for j in range(P)] for i in range(P)])

    true_dag_ = cd.DAG.from_amat(true_dag)
    true_cpdag = true_dag_.cpdag().to_amat()
    estimated_dag = cd.DAG.from_amat(est_)
    estimated_cpdag = estimated_dag.cpdag().to_amat()
    SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))

    skeleton_estimated, skeleton_true = skeleton(est_), skeleton(true_dag)
    SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
    TPR = np.sum(np.logical_and(est_, true_dag))/np.sum(true_dag)
    FPR = (np.sum(est_) - np.sum(np.logical_and(est_, true_dag)))/(P*P - np.sum(true_dag))
    print(f"TPR: {TPR}; FPR:{FPR}; shd_cpdag: {SHD_cpdag}; SHDs: {SHDs}; obj:{min_obj}; Time:{np.round(end_i - start_i, 3)}")
    # print(true_dag)



# np.sum(np.logical_and(moral_lasso.values, true_dag))/np.sum(true_dag)
# np.sum(np.logical_and(est_, true_dag))/np.sum(true_dag)
# np.sum(moral_lasso.values)
# np.sum(est_)