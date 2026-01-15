import causaldag as cd
# from causaldag.utils.ci_tests import gauss_ci_suffstat, gauss_ci_test
# from causaldag.utils.invariance_tests import gauss_invariance_test,
# gauss_ci_suffstat, gauss_ci_test,
from causaldag import partial_correlation_suffstat, partial_correlation_test, unknown_target_igsp, gauss_invariance_suffstat, MemoizedInvarianceTester, MemoizedCI_Tester
from utils import *

import numpy as np

# experiment setting
for graph in [1]:
    # graph = 2
    for lmbda in [0.01]: # for lmbda in [0.05, 0.06, 0.08, 0.1, 0.12]:
        for iter in range(1, 2):
            print(f"Iteration {iter}!!!!!!!!!!")
            datas, moral_graph, true_graph, interventions = read_data(graph, iter)

            n, p = datas[0].shape
            # estimated_gamma, estimated_delta, mipgap, time_i = optimization(datas, moral_graph, lmbda)
            # # estimated_gamma, estimated_delta = optimization(datas, moral_graph, 10*np.log(p)/np.sum([d.shape[0] for d in datas]))
            # estimated_B = np.array([[1 if estimated_gamma[i, j] != 0 and i != j else 0 for j in range(p)] for i in range(p)])
            # np.savetxt(f'./Results/estimations/micodag_dag_graph{graph}_lambda{lmbda}_iter{iter}.txt', estimated_B, fmt='%d', delimiter=',')
            # np.savetxt(f'./Results/estimations/micodag_intervention_targets_graph{graph}_lambda{lmbda}_iter{iter}.txt', estimated_delta, fmt='%d', delimiter=',')
            true_B = np.zeros((p, p))
            for idx in true_graph.values:
                true_B[idx[0] - 1, idx[1] - 1] = 1
            #
            #
            # # create a dag for computing shd for cpdag
            true_dag = cd.DAG.from_amat(np.array(true_B))
            true_cpdag = true_dag.cpdag().to_amat()



            obs_samples = datas[0]
            iv_samples_list = datas[1:]
            targets_list = interventions
            nodes = set(range(10))


            obs_suffstat = partial_correlation_suffstat(obs_samples)
            invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)

            # Create conditional independence tester and invariance tester
            alpha = 1e-3
            alpha_inv = 1e-3
            ci_tester = MemoizedCI_Tester(partial_correlation_suffstat, obs_suffstat, alpha=alpha)
            invariance_tester = MemoizedInvarianceTester(partial_correlation_test, invariance_suffstat, alpha=alpha_inv)

            # Run UT-IGSP
            setting_list = [dict(known_interventions=[]) for _ in targets_list]
            est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
            print(est_targets_list)