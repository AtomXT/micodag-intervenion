import numpy as np

from MICODAGCD import *
# data = bnlearn.bnlearn.import_example('sachs',,verbose=0)
int_data = pd.read_table('../Data/ReadData/sachs.interventional.txt')
obs_data = pd.read_table('../Data/ReadData/sachs.data.txt')
print(np.round(np.cov(obs_data.T),2))