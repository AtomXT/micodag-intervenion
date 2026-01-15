# 2024/04/13 Tong Xu
# Learning DAG from observational and interventional data.
# For each environment e, we have variables delta^e representing the DAG and the intervention targets.
# Instead of solving the formulation optimally, I want to use alternating optimization.
import time

from utils import *
from collections import deque
import pandas as pd
import causaldag as cd
from collections import defaultdict


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


def objective(Gamma, gammas, sigma_hats, deltas, weights, lam):
    P = len(Gamma)
    obj = 0
    for e, w in enumerate(weights):
        obj += w * sum([-2*np.log(Gamma[i, i]*(1-deltas[e][i]) + gammas[e][i]*deltas[e][i]) for i in range(P)])
        G = np.diag(gammas[e])
        obj += w * np.trace((Gamma@(np.eye(P)-np.diag(deltas[e]))@Gamma.T + G@np.diag(deltas[e])@G)@sigma_hats[e])
    obj += lam*(np.count_nonzero(Gamma)-P + sum([np.sum(deltas[e]) for e in range(len(weights))]))
    return obj

def A_tilde(Gamma, sigma_hat, delta, u, v):
    P = len(delta)
    return 2*sum([(1-delta[v])*Gamma[j, v]*sigma_hat[j, u] for j in range(P) if j != u])
    # return 2 * sum([Gamma[j, v] * sigma_hat[j, u] for j in range(P) if j != u])

def gamma_hat_uv(Gamma, sigma_hats, deltas, weights, u, v, lam):
    numerator = sum([w*A_tilde(Gamma, sigma_hats[e], deltas[e], u, v) for e, w in enumerate(weights)])
    # numerator = sum([w * (1 - deltas[e][v]) * A_tilde(Gamma, sigma_hats[e], deltas[e], u, v) for e, w in enumerate(weights)])
    denominator = 2*sum([(1-deltas[e][v])*w*sigma_hats[e][u, u] for e, w in enumerate(weights)])
    return -numerator/denominator if lam <= numerator*numerator/(2*denominator) else 0


def gamma_hat_uu(Gamma, sigma_hats, deltas, weights, u):
    star = sum([w*A_tilde(Gamma, sigma_hats[e], deltas[e], u, u) for e, w in enumerate(weights)])
    # star = sum([w * (1-deltas[e][u])*A_tilde(Gamma, sigma_hats[e], deltas[e], u, u) for e, w in enumerate(weights)])
    diag_sum = sum([w*sigma_hats[e][u, u] for e, w in enumerate(weights) if deltas[e][u] == 0])
    weight_sum = sum([w for e, w in enumerate(weights) if deltas[e][u] == 0])
    return (-star + np.sqrt(star*star + 16*diag_sum*weight_sum))/(4*diag_sum)


def solve_int(data_list, lam, gamma=None, moral=None, deltas=None, MAX_cycles=100):
    # interventional data only
    n_e = len(data_list)  # number of environment
    n_list = np.array([data_list[i].shape[0] for i in range(n_e)])  # list of sample size for each environment
    N = sum(n_list)  # total number of samples
    P = data_list[0].shape[1]  # number of nodes
    weights = n_list / N  # weights of each environment

    # moral
    if moral is None:
        pool_data = np.vstack(data_list)
        pool_data = (pool_data - pool_data.mean(axis=0))/pool_data.std(axis=0)
        moral = (abs(np.cov(pool_data.T)) > 0.001) + 0
        moral = moral - np.diag(np.diag(moral))

    # data process
    sigma_hats = [data_list[e].T @ data_list[e] / n_list[e] for e in range(n_e)]

    # initialization
    min_obj = np.inf
    objs = []
    Gamma0 = gamma if gamma is not None else np.eye(P)
    if deltas:
        known_targets = True
    else:
        deltas = [np.zeros(P) for e in range(n_e)]
        known_targets = False
    gammas = [1 / np.sqrt(np.diag(sigma_hats[e])) for e in range(n_e)]
    support_counter = defaultdict(int)

    for t in range(MAX_cycles):
        # Updating Gamma^e0
        for u in range(P):
            tmp = gamma_hat_uu(Gamma0, sigma_hats, deltas, weights, u)
            Gamma0[u, u] = tmp if tmp < np.inf else Gamma0[u, u]
            if np.isnan(Gamma0[u, u]):
                print("Wrong update function!!!!")
            for v in range(P):
                if moral[u, v] == 1:
                    temp_gamma = Gamma0.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    if cycle_uv:
                        Gamma0[u, v] = 0
                        if u == v:
                            print("setting diagonal to be zero")
                    else:
                        tmp = gamma_hat_uv(Gamma0, sigma_hats, deltas, weights, u, v, lam)
                        Gamma0[u, v] = tmp if tmp < np.inf else Gamma0[u, v]
        obj_t = objective(Gamma0, gammas, sigma_hats, deltas, weights, lam)
        if np.isnan(obj_t):
            print(f"Objective nan, Gamma0_diag: {np.diag(Gamma0)}")

        support_i = str(np.array(Gamma0 != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma0)):
                if u != v:
                    tmp = gamma_hat_uv(Gamma0, sigma_hats, deltas, weights, u, v, lam)
                    Gamma0[u, v] = tmp if tmp < np.inf else Gamma0[u, v]
                else:
                    tmp = gamma_hat_uu(Gamma0, sigma_hats, deltas, weights, u)
                    Gamma0[u, u] = tmp if tmp < np.inf else Gamma0[u, u]
            support_counter[support_i] = 0
            obj_t = objective(Gamma0, gammas, sigma_hats, deltas, weights, lam)

        if len(objs) > 1 and obj_t == objs[-1]:
            # if objs and objs[-1] - obj_t < 1e-6:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        objs.append(obj_t)
        min_obj = obj_t

        # Updating delta
        if not known_targets:
            for e in range(n_e):
                for i in range(P):
                    left = -2 * np.log(gammas[e][i]) + gammas[e][i] * gammas[e][i] * sigma_hats[e][i, i] + lam
                    right = -2 * np.log(Gamma0[i, i]) + sum(
                        [Gamma0[k, i] * Gamma0[j, i] * sigma_hats[e][j, k] for k in range(P) for j in range(P)])
                    deltas[e][i] = 1 if left < right else 0

    # print(objective(Gamma0, gammas, sigma_hats, deltas, weights, lam))
    # sum([w * (1 - deltas[e][0]) * (sigma_hats[e] @ Gamma)[0, 0] * Gamma[0, 0] for e, w in enumerate(weights)]) - sum([(1 - deltas[e][0]) * w for e, w in enumerate(weights)]) == 0
    # sum([w * (1 - deltas[e][2]) * (sigma_hats[e] @ Gamma)[0, 2] for e, w in enumerate(weights)])
    # sum([w*(1-deltas[e][u]) for u in range(10) for e,w in enumerate(weights)]) - sum([w * np.trace((Gamma@(np.eye(P)-np.diag(deltas[e]))@Gamma.T)@sigma_hats[e]) for e, w in enumerate(weights)])
    # sum([w * np.trace((Gamma0@(np.eye(P)-np.diag(deltas[e]))@Gamma0.T + np.diag(gammas[e])@np.diag(deltas[e])@np.diag(gammas[e]))@sigma_hats[e]) for e,w in enumerate(weights)])
    # np.log(np.linalg.det(sum([w * sigma_hats[e] for e, w in enumerate(weights)]))) - sum([w * np.log(np.linalg.det(sigma_hats[e])) for e, w in enumerate(weights)])
    # sum([w * -np.log(np.linalg.det((Gamma0@(np.eye(P)-np.diag(deltas[e]))@Gamma0.T + np.diag(gammas[e])@np.diag(deltas[e])@np.diag(gammas[e]))@sigma_hats[e])) for e,w in enumerate(weights)]) + np.log(np.linalg.det(np.sum([w*(Gamma0@(np.eye(P)-np.diag(deltas[e]))@Gamma0.T + np.diag(gammas[e])@np.diag(deltas[e])@np.diag(gammas[e]))@sigma_hats[e] for e,w in enumerate(weights)], axis=0)))
    return Gamma0, deltas, min_obj, objs
def solve(data_list,  lam, gamma=None, moral=None, deltas=None, MAX_cycles=100):
    n_e = len(data_list)  # number of environment
    n_list = np.array([data_list[i].shape[0] for i in range(n_e)]) # list of sample size for each environment
    N = sum(n_list)  # total number of samples
    P = data_list[0].shape[1]  # number of nodes
    weights = n_list / N  # weights of each environment

    # moral
    if moral is None:
        pool_data = np.vstack(data_list)
        pool_data = (pool_data - pool_data.mean(axis=0)) / pool_data.std(axis=0)
        moral = (abs(np.cov(pool_data.T)) > 0.001) + 0
        moral = moral - np.diag(np.diag(moral))

    # data process
    sigma_hats = [data_list[e].T @ data_list[e] / n_list[e] for e in range(n_e)]

    # initialization
    min_obj = np.inf
    objs = []
    Gamma0 = gamma if gamma is not None else np.eye(P)
    if deltas:
        known_targets = True
    else:
        deltas = [np.zeros(P) for e in range(n_e)]
        known_targets = False
    gammas = [1/np.sqrt(np.diag(sigma_hats[e])) for e in range(n_e)]
    support_counter = defaultdict(int)



    for t in range(MAX_cycles):
        # Updating Gamma^e0
        for u in range(P):
            Gamma0[u, u] = gamma_hat_uu(Gamma0, sigma_hats, deltas, weights, u)
            for v in range(P):
                if u!=v and moral[u, v] == 1:
                    temp_gamma = Gamma0.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    if cycle_uv:
                        Gamma0[u, v] = 0
                    else:
                        Gamma0[u, v] = gamma_hat_uv(Gamma0, sigma_hats, deltas, weights, u, v, lam)
        obj_t = objective(Gamma0, gammas, sigma_hats, deltas, weights, lam)

        support_i = str(np.array(Gamma0 != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma0)):
                if u != v:
                    Gamma0[u, v] = gamma_hat_uv(Gamma0, sigma_hats, deltas, weights, u, v, lam)
                else:
                    Gamma0[u, u] = gamma_hat_uu(Gamma0, sigma_hats, deltas, weights, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma0, gammas, sigma_hats, deltas, weights, lam)

        if len(objs) > 1 and obj_t == objs[-1]:
        # if objs and objs[-1] - obj_t < 1e-6:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        objs.append(obj_t)
        min_obj = obj_t

        # Updating delta
        if not known_targets:
            for e in range(1, n_e):
                for i in range(P):
                    left = -2*np.log(gammas[e][i]) + gammas[e][i]*gammas[e][i]*sigma_hats[e][i,i] + lam
                    right = -2*np.log(Gamma0[i,i]) + sum([Gamma0[k, i]*Gamma0[j, i]*sigma_hats[e][j, k] for k in range(P) for j in range(P)])
                    deltas[e][i] = 1 if left < right else 0



    # print(objective(Gamma0, gammas, sigma_hats, deltas, weights, lam))
    # sum([w * (1 - deltas[e][0]) * (sigma_hats[e] @ Gamma)[0, 0] * Gamma[0, 0] for e, w in enumerate(weights)]) - sum([(1 - deltas[e][0]) * w for e, w in enumerate(weights)]) == 0
    # sum([w * (1 - deltas[e][2]) * (sigma_hats[e] @ Gamma)[0, 2] for e, w in enumerate(weights)])
    # sum([w*(1-deltas[e][u]) for u in range(10) for e,w in enumerate(weights)]) - sum([w * np.trace((Gamma@(np.eye(P)-np.diag(deltas[e]))@Gamma.T)@sigma_hats[e]) for e, w in enumerate(weights)])
    # sum([w * np.trace((Gamma0@(np.eye(P)-np.diag(deltas[e]))@Gamma0.T + np.diag(gammas[e])@np.diag(deltas[e])@np.diag(gammas[e]))@sigma_hats[e]) for e,w in enumerate(weights)])
    # np.log(np.linalg.det(sum([w * sigma_hats[e] for e, w in enumerate(weights)]))) - sum([w * np.log(np.linalg.det(sigma_hats[e])) for e, w in enumerate(weights)])
    # sum([w * -np.log(np.linalg.det((Gamma0@(np.eye(P)-np.diag(deltas[e]))@Gamma0.T + np.diag(gammas[e])@np.diag(deltas[e])@np.diag(gammas[e]))@sigma_hats[e])) for e,w in enumerate(weights)]) + np.log(np.linalg.det(np.sum([w*(Gamma0@(np.eye(P)-np.diag(deltas[e]))@Gamma0.T + np.diag(gammas[e])@np.diag(deltas[e])@np.diag(gammas[e]))@sigma_hats[e] for e,w in enumerate(weights)], axis=0)))
    return Gamma0, deltas, min_obj, objs

def solve_auto(data_list, lam=None, moral=None, deltas=None, MAX_cycles=100):
    lam = np.linspace(0, 0.1, 10) if lam is None else lam
    Gammas = []
    Gamma0, deltas, min_obj, objs = solve(data_list, lam[0], gamma=None, moral=moral, deltas=deltas, MAX_cycles=100)
    Gammas.append(Gamma0)
    for l in lam[1:]:
        Gamma0, deltas, min_obj, objs = solve(data_list, l, gamma=Gamma0.copy(), moral=moral, deltas=deltas, MAX_cycles=100)
        Gammas.append(Gamma0)
    return Gammas

def solve_int_auto(data_list, lam=None, moral=None, deltas=None, MAX_cycles=100):
    lam = np.linspace(0, 0.1, 10) if lam is None else lam
    Gammas = []
    Gamma0, deltas, min_obj, objs = solve_int(data_list, lam[0], gamma=None, moral=moral, deltas=deltas, MAX_cycles=100)
    Gammas.append(Gamma0)
    for l in lam[1:]:
        Gamma0, deltas, min_obj, objs = solve_int(data_list, l, gamma=Gamma0.copy(), moral=moral, deltas=deltas, MAX_cycles=100)
        Gammas.append(Gamma0)
    return Gammas




if __name__ == '__main__':
    m_list = np.array([10, 20, 50, 100, 200])
    lmbdas = np.around(3*np.log(m_list)/(2*6*m_list),2)
    micodag_int_results = []
    for graph in [2]:
        # graph = 2
        for lmbda in [lmbdas[graph-1]]:  # for lmbda in [0.05, 0.06, 0.08, 0.1, 0.12]:
            # lmbda = 0.1
            times1 = []
            times2 = []
            for iter in range(1, 2):
                print(f"Iteration {iter}!!!!!!!!!!")
                datas, moral_graph, true_graph, interventions = read_data(graph, iter)

                n, p = datas[0].shape
                true_moral_ = np.array([[0] * p for i in range(p)])
                for i in range(len(moral_graph)):
                    true_moral_[moral_graph.iloc[i, 0] - 1][moral_graph.iloc[i, 1] - 1] = 1
                one = np.ones((20, 20))
                one = one - np.diag(np.diag(one))
                start = time.time()
                estimated_gamma, estimated_delta, min_obj = solve(datas, one, lmbda)
                end = time.time()
                # estimated_gamma, estimated_delta = optimization(datas, moral_graph, 10*np.log(p)/np.sum([d.shape[0] for d in datas]))
                estimated_B = np.array(
                    [[1 if estimated_gamma[i, j] != 0 and i != j else 0 for j in range(p)] for i in range(p)])
                # np.savetxt(f'./Results/estimations/micodagcd_dag_graph{graph}_lambda{lmbda}_iter{iter}.txt', estimated_B,
                #            fmt='%d', delimiter=',')
                # np.savetxt(
                #     f'./Results/estimations/micodagcd_intervention_targets_graph{graph}_lambda{lmbda}_iter{iter}.txt',
                #     estimated_delta, fmt='%d', delimiter=',')
                true_B = np.zeros((p, p))
                for idx in true_graph.values:
                    true_B[idx[0] - 1, idx[1] - 1] = 1
                #
                #
                # # create a dag for computing shd for cpdag
                true_dag = cd.DAG.from_amat(np.array(true_B))
                true_cpdag = true_dag.cpdag().to_amat()
                estimated_dag = cd.DAG.from_amat(np.array(estimated_B))
                estimated_cpdag = estimated_dag.cpdag().to_amat()
                SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
                print(f"The d_cpdag is {SHD_cpdag}, it took {round(end-start,2)} seconds, and the obj is {min_obj}.")
                micodag_int_results.append([p, lmbda, iter, SHD_cpdag, end-start])
    micodag_int_results = pd.DataFrame(micodag_int_results, columns=['m', 'lambda', 'iter', 'd_cpdag', 'time'])
    print(micodag_int_results)
    # micodag_int_results.to_csv('./Results/micodag_int_results.csv', index=False)
    print(f"The averaged d_cpdag is {np.mean(micodag_int_results['d_cpdag'])}.")
    print(f"The averaged time is {np.mean(micodag_int_results['time'])}.")
                # print("intervention targets estimation", estimated_delta)

