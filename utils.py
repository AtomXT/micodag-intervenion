import numpy as np
import pandas as pd
import os


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





