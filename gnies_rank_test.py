from utils import *
import time
import causaldag as cd
import gnies



m_list = np.array([10, 20, 50, 100, 200])
# experiment setting
for graph in [1,2,3]:
    # graph = 2
    for lmbda in [6]: # for lmbda in [0.05, 0.06, 0.08, 0.1, 0.12]:
        gnies_results = []
        for iter in range(1, 11):
            print(f"Iteration {iter}!!!!!!!!!!")
            datas, moral_graph, true_graph, interventions = read_data(graph, iter)
            n, p = datas[0].shape
            true_B = np.zeros((p, p))
            for idx in true_graph.values:
                true_B[idx[0] - 1, idx[1] - 1] = 1
            #
            # # create a dag for computing shd for cpdag
            true_dag = cd.DAG.from_amat(np.array(true_B))
            true_cpdag = true_dag.cpdag().to_amat()

            # #####################
            data_gnies = [data_i.values for data_i in datas]
            start = time.time()
            approach = 'rank'
            gnies_estimate = gnies.fit(data_gnies, approach=approach, lmbda=lmbda)
            end = time.time()
            gnies_results.append([m_list[graph-1], lmbda, iter, end - start])
            gnies_estimated_ess = gnies_estimate[1]
            np.savetxt(f'./Results/estimations/gnies_{approach}_essential_graph{graph}_iter{iter}.txt', gnies_estimated_ess, fmt='%d', delimiter=',')
            gnies_estimated_targets = np.array(list(gnies_estimate[2]))
            np.savetxt(f'./Results/estimations/gnies_{approach}_intervention_targets_graph{graph}_iter{iter}.txt', gnies_estimated_targets, fmt='%d', delimiter=',')
            gnies_SHD_cpdag = np.sum(np.abs(gnies_estimated_ess - true_cpdag[0]))
            print(gnies_SHD_cpdag, f"gnies uses time: {end-start} seconds.")
            print("intervention targets estimation", gnies_estimated_targets)

            print("--------------")
# print(times2)
gnies_rank_results = pd.DataFrame(gnies_results, columns=['m', 'lambda', 'iteration', 'time'])
print(gnies_rank_results)
gnies_rank_results.to_csv(f'./Results/gnies_rank_lambda{lmbda}_time.csv')

# 4  3  4 13  6  4  6  5  6  8 results for lambda = 0.1