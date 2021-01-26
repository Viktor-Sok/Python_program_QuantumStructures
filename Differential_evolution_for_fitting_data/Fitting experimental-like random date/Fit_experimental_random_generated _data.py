import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import timeit
from scipy.ndimage import gaussian_filter1d

"""|||||||||||||||||Loading data to work with from  files in the .dat format||||||||||||||"""
#Read in the experimental random generated isotropic signal 
data = pd.read_table('Random_Data_experiment_Sigma_11.7.dat',\
    header = None)
npData = data.to_numpy()

#Read in an Instrumental Respond Function(IRF)
IRF_data = pd.read_table('IRFy_wm_free.dat',\
    header = None)
npIRF = IRF_data.to_numpy()
"""|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"""


"""==============Some of the important global variables===================="""
I_iso = npData[:,1]
dt = (npIRF[1,0] - npIRF[0,0]) #interval for equally spaced data in time domain
IRF = npIRF[:,1]/(np.sum(npIRF[:,1])*dt) # normalizing IRF function
t = npData[:,0]
sigma2 = np.maximum(I_iso,np.ones(len(I_iso))) #removing all of the values which are less than one count
"""========================================================================"""
plt.figure(5)
plt.plot(t, I_iso,'.',color ='r',markersize = 2)
plt.plot(t, npData[:,2],linewidth = 0.5, color = 'b')


"""++++++++++++++++Fix boundaries of a search area down here++++++++++++++++"""
bounds = [(0,5000),#T (Shift of the model function to the IRF in time domain)  
          (0, 100), #a0(free term in the  model function)
          (0, 1000),#a1
          (0,1000),  #a2
          (0,1000),  #a3
          (0, 5000), #a4
          (0,5000),#t1
          (0,5000), #t2
          (0,5000),   #t3
          (0,5000)   #t4
         ]
"""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""


"""---------Define model decay function here--------------"""
def model_decay_function(T,a0,a1,a2,a3,a4,t1,t2,t3,t4):
    #assumed function of the isotropic part of a fluorescence decay
    return a0 + a1*np.exp(-t/t1)+a2*np.exp(-t/t2)+a3*np.exp(-t/t3) + a4*np.exp(-t/t4), int(round(T/dt));
"""-------------------------------------------------------"""



def observable_decay_function(*parameters):
    #calculation of observable in the experiment decay function by means of convolution through fft
    model_function, shiftT = model_decay_function(*parameters)
    fourier_I_model = np.fft.fft(model_function)
    IRF_shifted = np.roll(IRF,shiftT)
    fourier_IRF_shifted = np.fft.fft(IRF_shifted)
    return dt * np.fft.ifft(fourier_I_model * fourier_IRF_shifted).real


"""============ Choose one of the objective functions to minimize=================="""
def least_square_roots(parameters):
    #objective function to minimize
    return np.sum((I_iso - observable_decay_function(*parameters))**2)

def max_likelyhood_estimator(parameters):
    #objective function to minimize
    return 2*np.sum(observable_decay_function(*parameters) - sigma2) - \
           2*np.sum(sigma2*np.log(observable_decay_function(*parameters)/sigma2))
"""================================================================================="""



def finding_the_minimum():
    #using diffirential evolution to minimize objective function
    start = timeit.default_timer()
    result = differential_evolution(max_likelyhood_estimator, bounds,  mutation = 0.5, recombination = 0.3, popsize = 10 )
    stop = timeit.default_timer()
    execution_time = stop - start
    print("Program Executed in "+str(execution_time))
    return result

result = finding_the_minimum()
parameters = result.x
print(result.success)
print(result.message)
print(parameters)
plt.figure(1)
plt.plot(t,abs(npData[:,2] - observable_decay_function(*parameters)) )
plt.figure(2)    
plt.plot(t, I_iso - observable_decay_function(*parameters), '.',markersize=5)
plt.figure(3)
plt.plot(t, I_iso,'.', color = 'r',markersize=2)
plt.plot(t,observable_decay_function(*parameters), linewidth = 0.5, color = 'b')
plt.figure(4)
n, bins, patches = plt.hist(I_iso - observable_decay_function(*parameters), 50, density=True, facecolor='g', alpha=0.75)
plt.show()
