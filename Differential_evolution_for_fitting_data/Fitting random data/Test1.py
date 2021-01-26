import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import timeit
#Read in experimental data
experiment_data = pd.read_table('Test.dat',\
    header = None, usecols =[1])
#print(experiment_data)
npData = experiment_data.to_numpy().transpose()
a = np.array([2,4,6,8])
print(a)
k = np.log(-2)
print(k)
