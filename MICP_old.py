import gurobipy as gp
from gurobipy import GRB
from utils import *
import time
import causaldag as cd


def optimization(data, moral, l):
    # data[0] is the observational dataset, others are interventional
    n_environment = len(data)  # number of environments
    n_samples = [data[i].shape[0] for i in range(n_environment)]  # list of the number of samples in each environment
    p = data[0].shape[1]  # number of nodes of this graph
    list_edges = []
    for edge in moral.values:
        list_edges.append((edge[0] - 1, edge[1] - 1))
        list_edges.append((edge[1] - 1, edge[0] - 1))
    E = [(i, j) for i in range(p) for j in range(p) if i != j]
    non_edges = list(set(E) - set(list_edges))
    Sigma_hats = [data[i].values.T @ data[i].values / n_samples[i] for i in range(n_environment)]

    model = gp.Model()
    Gammas = [{} for _ in range(n_environment)]
    for e in range(n_environment):
        for i in range(p):
            for j in range(p):
                if i == j:
                    Gammas[e][i, j] = model.addVar(lb=1e-5, vtype=GRB.CONTINUOUS)
                else:
                    Gammas[e][i, j] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    psi = model.addMVar((p, 1), lb=1, ub=p, vtype=GRB.CONTINUOUS)
    # psis = [model.addMVar((p, 1), lb=1, ub=p, vtype=GRB.CONTINUOUS, name='psi') for e in range(n_environment)]
    # gs = [{} for _ in range(n_environment)]
    # for e in range(n_environment):
    #     for i in range(p):
    #         for j in range(p):
    #             gs[e][i, j] = model.addVar(vtype=GRB.BINARY)

    g = {}
    for i in range(p):
        for j in range(p):
            g[i, j] = model.addVar(vtype=GRB.BINARY)

    deltas = [{} for _ in range(n_environment)]
    for e in range(1, n_environment):
        for i in range(p):
            deltas[e][i] = model.addVar(vtype=GRB.BINARY)

    # Variables for outer approximation
    Ts = [{} for _ in range(n_environment)]
    for e in range(n_environment):
        for i in range(p):
            Ts[e][i] = model.addVar(lb=-10, ub=100, vtype=GRB.CONTINUOUS)
            # This gives Gamma[e][i,i] a range about [0.0001, 100]

    zetas = [{} for _ in range(n_environment)]
    for e in range(1, n_environment):
        for i in range(p):
            for j in range(p):
                zetas[e][i, j] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    xis = [{} for _ in range(n_environment)]
    for e in range(1, n_environment):
        for i in range(p):
            xis[e][i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    gamma = [{} for _ in range(n_environment)]
    for e in range(1, n_environment):
        for i in range(p):
            gamma[e][i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    model._Ts = Ts
    model._Gammas = Gammas
    # model._gs = gs

    # define the callback function
    def logarithmic_callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # Get the value of Gamma
            for e in range(n_environment):
                Gamma_val = model.cbGetSolution(model._Gammas[e])
                for i in range(p):
                    model.cbLazy(model._Ts[e][i] >= -2 * np.log(Gamma_val[i, i]) - 2 / Gamma_val[i, i] * (
                            model._Gammas[e][i, i] - Gamma_val[i, i]))

    log_term = 0
    for e in range(n_environment):
        for i in range(p):
            # log_term += Ts[e][i]
            log_term += Ts[e][i] * n_samples[e]/np.sum(n_samples)
    trace = gp.QuadExpr()
    for e in range(n_environment):
        for k in range(p):
            for j in range(p):
                for i in range(p):
                    # trace += Gammas[e][i, k] * Gammas[e][i, j] * Sigma_hats[e][j, k]
                    trace += Gammas[e][k, i]*Gammas[e][j, i]*Sigma_hats[e][j, k] * n_samples[e]/np.sum(n_samples)
    penalty = gp.LinExpr()
    for i, j in E:
        penalty += l*g[i, j]*n_environment
    penalty += l*gp.quicksum(gp.quicksum(deltas[e][i] for i in range(p)) for e in range(1, n_environment))

    model.setObjective(log_term + trace + penalty, GRB.MINIMIZE)

    # solve the problem without constraints to get big_M
    model.Params.lazyConstraints = 1
    model.Params.OutputFlag = 0
    model.optimize(logarithmic_callback)
    Big_M_obj = model.ObjVal
    model.update()

    big_M = 0
    for e in range(n_environment):
        for j, k in list_edges:
            big_M = max(big_M, abs(Gammas[e][j, k].x))
    M = 2*big_M

    model.addConstrs(Gammas[0][i, i] <= M for i in range(p))
    model.addConstrs(Gammas[0][j, k] <= M * g[j, k] for j, k in list_edges)
    model.addConstrs(Gammas[0][j, k] >= -M * g[j, k] for j, k in list_edges)
    model.addConstrs(1 - p + p * g[j, k] <= psi[k] - psi[j] for j, k in list_edges)
    model.addConstrs(Gammas[0][j, k] == 0 for j, k in non_edges)  # Use moral structure
    model.addConstrs(g[j, k] == 0 for j, k in non_edges)  # Use moral structure

    # This set of constraints does not use linearizations.
    for e in range(1, n_environment):
        for i in range(p):
            for j in range(p):
                if i != j:
                    # model.addConstr(Gammas[e][i, j] == Gammas[0][i, j]*(1-deltas[e][j]))
                    model.addConstr(Gammas[e][i, j] <= (Gammas[0][i, j]+0.1) * (1 - deltas[e][j]))
                    model.addConstr(Gammas[e][i, j] >= (Gammas[0][i, j]-0.1) * (1 - deltas[e][j]))
                else:
                    model.addConstr(Gammas[e][j, j] == Gammas[0][j, j]*(1-deltas[e][j]) + gamma[e][j]*deltas[e][j])


    # # This set of constraints set all Gammas to be the same. I use them to test whether we can get the same performance
    # # as micodag.
    # for e in range(1, n_environment):
    #     for i in range(p):
    #         for j in range(p):
    #             model.addConstr(Gammas[e][i, j] == Gammas[0][i, j])


    # for e in range(1, n_environment):
    #     for j in range(p):
    #         model.addConstr(xis[e][j] >= -M * deltas[e][j])
    #         model.addConstr(xis[e][j] <= M * deltas[e][j])
    #         model.addConstr(gamma[e][j] - xis[e][j] >= -M * (1 - deltas[e][j]))
    #         model.addConstr(gamma[e][j] - xis[e][j] <= M * (1 - deltas[e][j]))
    #
    # for e in range(1, n_environment):
    #     for i in range(p):
    #         for j in range(p):
    #             model.addConstr(zetas[e][i, j] >= -M * (1 - deltas[e][j]))
    #             model.addConstr(zetas[e][i, j] <= M * (1 - deltas[e][j]))
    #             model.addConstr(Gammas[0][i, j] - zetas[e][i, j] >= -M * deltas[e][j])
    #             model.addConstr(Gammas[0][i, j] - zetas[e][i, j] <= M * deltas[e][j])
    #             if i == j:
    #                 model.addConstr(Gammas[e][j, j] == zetas[e][j, j] + xis[e][j])
    #             else:
    #                 model.addConstr(Gammas[e][i, j] == zetas[e][i, j])
    #
    # # force the delta to be the true targets
    # for e in range(1, n_environment):
    #     intervention = interventions[e-1]
    #     for j in range(1, p+1):
    #         if j in intervention:
    #             model.addConstr(deltas[e][j-1] == 1)
    #         else:
    #             model.addConstr(deltas[e][j-1] == 0)


    # Solve
    model.Params.TimeLimit = 50*p
    model.Params.lazyConstraints = 1
    model.Params.OutputFlag = 0
    start = time.time()
    model.optimize(logarithmic_callback)
    end = time.time()

    Gamma_opt = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            Gamma_opt[i, j] = Gammas[0][i, j].x
    Gamma_opt = Gamma_opt
    Gammas_opt = [np.reshape([Gammas[e][i, j].x for i in range(p) for j in range(p)], (p, p)) for e in range(n_environment)]
    delta_opt = []
    for e in range(1, n_environment):
        delta_opt.extend([i+1 for i in range(p) if abs(deltas[e][i].x) > 0.5])
    delta_opt = list(set(delta_opt))
    return Gamma_opt, delta_opt, model.MIPGAP, end-start


# experiment setting
for graph in [1]:
    # graph = 2
    for lmbda in [0.002]: # for lmbda in [0.05, 0.06, 0.08, 0.1, 0.12]:
    # lmbda = 0.1
        times1 = []
        times2 = []
        micodag_int_results = []
        for iter in range(1, 11):
            print(f"Iteration {iter}!!!!!!!!!!")
            datas, moral_graph, true_graph, interventions = read_data(graph, iter)

            n, p = datas[0].shape
            estimated_gamma, estimated_delta, mipgap, time_i = optimization(datas, moral_graph, lmbda)
            # estimated_gamma, estimated_delta = optimization(datas, moral_graph, 10*np.log(p)/np.sum([d.shape[0] for d in datas]))
            estimated_B = np.array([[1 if estimated_gamma[i, j] != 0 and i != j else 0 for j in range(p)] for i in range(p)])
            np.savetxt(f'./Results/estimations/micodag_dag_graph{graph}_lambda{lmbda}_iter{iter}.txt', estimated_B, fmt='%d', delimiter=',')
            np.savetxt(f'./Results/estimations/micodag_intervention_targets_graph{graph}_lambda{lmbda}_iter{iter}.txt', estimated_delta, fmt='%d', delimiter=',')
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
            micodag_int_results.append([iter, mipgap, time_i])
            print(SHD_cpdag)
            print("intervention targets estimation", estimated_delta)

            # #####################
            # from utils import *
            # import micodag as mic
            # # iter = 1
            # # graph = 1
            # datas, moral, true_graph, interventions = read_data(graph, iter)
            # data = np.concatenate(datas)
            # est_micodag = mic.optimize(pd.DataFrame(data), moral, 0.04)
            # est_B_micodag =  np.array([[1 if est_micodag[1][i, j] != 0 else 0 for j in range(10)] for i in range(10)])
            # estimated_dag_micodag = cd.DAG.from_amat(np.array(est_B_micodag))
            # estimated_cpdag_micodag = estimated_dag_micodag.cpdag().to_amat()
            # SHD_cpdag = np.sum(np.abs(estimated_cpdag_micodag[0] - true_cpdag[0]))
            # print(SHD_cpdag)
        #
        #     data_gnies = [data_i.values for data_i in datas]
        #     start = time.time()
        #     approach = 'rank'
        #     gnies_estimate = gnies.fit(data_gnies, approach=approach)
        #     end = time.time()
        #     times1.append([iter, end - start])
        #     gnies_estimated_ess = gnies_estimate[1]
        #     # np.savetxt(f'./Results/estimations/gnies_{approach}_essential_graph{graph}_iter{iter}.txt', gnies_estimated_ess, fmt='%d', delimiter=',')
        #     gnies_estimated_targets = np.array(list(gnies_estimate[2]))
        #     # np.savetxt(f'./Results/estimations/gnies_{approach}_intervention_targets_graph{graph}_iter{iter}.txt', gnies_estimated_targets, fmt='%d', delimiter=',')
        #     gnies_SHD_cpdag = np.sum(np.abs(gnies_estimated_ess - true_cpdag[0]))
        #     print(gnies_SHD_cpdag, f"gnies uses time: {end-start} seconds.")
        #     print("intervention targets estimation", gnies_estimated_targets)
        #
        #
        #     start = time.time()
        #     approach = 'greedy'
        #     gnies_estimate = gnies.fit(data_gnies, approach=approach)
        #     end = time.time()
        #     times2.append([iter, end - start])
        #     gnies_estimated_ess = gnies_estimate[1]
        #     # np.savetxt(f'./Results/estimations/gnies_{approach}_essential_graph{graph}_iter{iter}.txt', gnies_estimated_ess, fmt='%d', delimiter=',')
        #     gnies_estimated_targets = np.array(list(gnies_estimate[2]))
        #     # np.savetxt(f'./Results/estimations/gnies_{approach}_intervention_targets_graph{graph}_iter{iter}.txt', gnies_estimated_targets, fmt='%d', delimiter=',')
        #     gnies_SHD_cpdag = np.sum(np.abs(gnies_estimated_ess - true_cpdag[0]))
        #     print(gnies_SHD_cpdag, f"gnies uses time: {end-start} seconds.")
        #     print("intervention targets estimation", gnies_estimated_targets)
        #     print("--------------")
        # print(times1)
        # print(times2)
        # gnies_rank_results = pd.DataFrame(times1, columns=['iteration', 'time'])
        # gnies_greedy_results = pd.DataFrame(times2, columns=['iteration', 'time'])
        micodag_int_results = pd.DataFrame(micodag_int_results, columns=['iteration', 'mipgap', 'time'])
        print(micodag_int_results)
        # gnies_rank_results.to_csv(f'./Results/estimations/gnies_rank_graph{graph}_lambda{lmbda}_time.csv')
        # gnies_greedy_results.to_csv(f'./Results/estimations/gnies_greedy_graph{graph}_lambda{lmbda}_time.csv')
        # micodag_int_results.to_csv(f'./Results/estimations/micodag_graph{graph}_lambda{lmbda}_mipgap_time.csv')

# 4  3  4 13  6  4  6  5  6  8 results for lambda = 0.1