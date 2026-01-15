from MICP import *
from MICODAGCD import *
import matplotlib.pyplot as plt


# redefine the data read function because the small graphs are stored in a different folder
def read_data(idx, n, iter):
    path = './Data/SyntheticData/Small/graph' + str(idx)
    environments = [file for file in os.listdir(path) if file.startswith("environment")]
    data = []
    data_i = pd.read_csv(path + f'/observational/data_n_{n}_iter_{iter}.csv', header=None)
    data.append(data_i)
    for env in environments:
        data_i = pd.read_csv(path+f'/{env}/data_n_{n}_iter_{iter}.csv', header=None)
        data.append(data_i)
    p = data[0].shape[1]
    moral = pd.read_table(f'./Data/SyntheticData/Moral_{p}.txt', header=None, sep=' ')
    true_dag = pd.read_table(f'./Data/SyntheticData/DAG_{p}.txt', header=None, sep=' ')
    with open(path+'/intervention_targets.txt', 'r') as file:
        lines = file.readlines()
    interventions = [list(map(int, line.strip().split(','))) for line in lines]
    return data, moral, true_dag, interventions


graph = 1
iter = 1
# n_list = [10, 20, 50, 100, 200]
n_list = [100]
p = 10
lmbda = 0.01
micodag_int_results = []
micodag_int_cd_results = []
for n in n_list:
    for i in range(1, 11):
        print(f"##############Running iteration {i} with n = {n}.##############")
        datas, moral_graph, true_graph, interventions = read_data(graph, n, iter)
        estimated_gamma, estimated_delta, mipgap, min_obj, time_i = optimization(datas, moral_graph, lmbda)
        estimated_B = np.array([[1 if estimated_gamma[i, j] != 0 and i != j else 0 for j in range(p)] for i in range(p)])
        # np.savetxt(f'./Results/estimations/micodag_dag_graph{graph}_lambda{lmbda}_iter{iter}.txt', estimated_B, fmt='%d', delimiter=',')
        # np.savetxt(f'./Results/estimations/micodag_intervention_targets_graph{graph}_lambda{lmbda}_iter{iter}.txt', estimated_delta, fmt='%d', delimiter=',')
        true_B = np.zeros((p, p))
        for idx in true_graph.values:
            true_B[idx[0] - 1, idx[1] - 1] = 1
        true_dag = cd.DAG.from_amat(np.array(true_B))
        true_cpdag = true_dag.cpdag().to_amat()
        estimated_dag = cd.DAG.from_amat(np.array(estimated_B))
        estimated_cpdag = estimated_dag.cpdag().to_amat()
        SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
        micodag_int_results.append([p, n, iter, lmbda, mipgap, min_obj, SHD_cpdag, time_i])
        print(SHD_cpdag)
        print("intervention targets estimation", estimated_delta)

        true_moral_ = np.array([[0] * p for i in range(p)])
        for i in range(len(moral_graph)):
            true_moral_[moral_graph.iloc[i, 0] - 1][moral_graph.iloc[i, 1] - 1] = 1
        start = time.time()
        estimated_gamma, estimated_delta, min_obj = solve(datas, true_moral_, lmbda)
        end = time.time()
        estimated_B = np.array(
            [[1 if estimated_gamma[i, j] != 0 and i != j else 0 for j in range(p)] for i in range(p)])
        true_dag = cd.DAG.from_amat(np.array(true_B))
        true_cpdag = true_dag.cpdag().to_amat()
        estimated_dag = cd.DAG.from_amat(np.array(estimated_B))
        estimated_cpdag = estimated_dag.cpdag().to_amat()
        SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
        print(f"The d_cpdag is {SHD_cpdag}, it took {round(end - start, 2)} seconds, and the obj is {min_obj}.")
        micodag_int_cd_results.append([p, n, iter, lmbda, min_obj, SHD_cpdag, end - start])

micodag_int_results = pd.DataFrame(micodag_int_results, columns=['p', 'n', 'iter', 'lmbda', 'gap', 'obj', 'cpdag', 'time'])
micodag_int_cd_results = pd.DataFrame(micodag_int_cd_results, columns=['p', 'n', 'iter', 'lmbda', 'obj', 'cpdag', 'time'])
# micodag_int_results.to_csv('./Results/Small_micodag_int_results.csv', index=False)
# micodag_int_cd_results.to_csv('./Results/Small_micodag_int_cd_results.csv', index=False)
print(micodag_int_results)
print(micodag_int_cd_results)

# n_list = [20, 50, 100, 200]
# results = pd.read_csv('./Results/Small_micodag_int_results.csv')
# results_cd = pd.read_csv('./Results/Small_micodag_int_cd_results.csv')
# f = plt.figure()
# plt.plot(n_list, results_cd.groupby(by='n').mean().obj[1:])
# plt.plot(n_list, results.groupby(by='n').mean().obj[1:])
# plt.xlabel('Number of samples per environment')
# plt.ylabel("Objective")
# plt.legend(['CD', 'MICODAG'])
# plt.show()
# f.savefig('./Results/plots/Small_graph.pdf')