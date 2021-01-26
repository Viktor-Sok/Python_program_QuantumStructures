import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import timeit
#Read in experimental data
experiment_data = pd.read_table('Random_Data_Sigma_0.2.dat',\
    header = None)

#Global variables
npData = experiment_data.to_numpy().transpose()
t = npData[0,:]
sigma2 = np.maximum(npData[1,:],np.ones(len(npData[1,:])))
bounds = [(1., 10.), #a0
          (10., 30.), #a1
          (1.,20.),  #a2
          (1.,10.),#a3
          #(1.,30.), #a4
          (50.,150.),#t1
          (20.,80.),#t2
          (1.,20.) #t3
          #(1,30.) #t4
         ]

def model_function(a0,a1,a2,a3,t1,t2,t3):
    #assumed function of physical process
    return a0 + a1*np.exp(-t/t1)+a2*np.exp(-t/t2)+a3*np.exp(-t/t3)

"""================================================================================
=============== Choose one of the objective functions to minimize===================
"""
def least_square_roots1(parameters):
    #objective function to minimize
    return np.sum((npData[1,:] - model_function(*parameters))**2)

def least_square_roots2(parameters):
    #objective function to minimize
    return np.sum((npData[1,:] - model_function(*parameters))**2/sigma2)

def max_likelyhood_estimator1(parameters):
    #objective function to minimize
    return 2*np.sum(model_function(*parameters) - npData[1,:]) - \
           2*np.sum(npData[1,:]*np.log(model_function(*parameters)/npData[1,:]))

def max_likelyhood_estimator2(parameters):
    #objective function to minimize
    return 2*np.sum((model_function(*parameters) +npData[1,:]*np.log(npData[1,:])) \
                    -npData[1,:]*(1+np.log(model_function(*parameters))))

"""===============================================================================
==================================================================================
"""

def finding_the_minimum():
    #using diffirential evolution to minimize objective function
    start = timeit.default_timer()
    result = differential_evolution(max_likelyhood_estimator1 ,bounds,  mutation = 0.5, recombination = 0.3, popsize = 10 )
    #result = differential_evolution(max_likelyhood_estimator2,bounds)
    stop = timeit.default_timer()
    execution_time = stop - start
    print("Program Executed in "+str(execution_time))
    return result.x


parameters = finding_the_minimum()
print(parameters)
plt.figure(1)    
plt.plot(t, npData[1,:] - model_function(*parameters))
plt.figure(2)
plt.plot(t, npData[2,:] - model_function(*parameters))
plt.show()
