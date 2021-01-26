import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import timeit

"""|||||||||||||||||Loading data to work with from  files in the .dat format||||||||||||||"""
#Read in the experimental data
experiment_data = pd.read_table('2020-11-05 PBS ADH_2mg_ml NADH_40uM 720nm  250mW Sh9 ZET436_10 2000s ZYLZYZYZ_exp.dat',\
    header = None, usecols = [0,1,2,3,4])
npData = experiment_data.to_numpy()

#Read in an Instrumental Respond Function(IRF)
IRF_data = pd.read_table('IRFy_wm_free.dat',\
    header = None)
npIRF = IRF_data.to_numpy()
"""|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"""


"""==============Some of the important global variables===================="""
dt = (npIRF[1,0] - npIRF[0,0]) #interval for equally spaced data in time domain
IRF = npIRF[:,1]/(np.sum(npIRF[:,1])*dt) # normalizing IRF function
t = npData[:,0]*1000
plt.figure(5)
plt.plot(t[0:501], IRF[0:501],'.',markersize=2)
"""========================================================================"""


def calculate_isotropic_part(darkX,darkY,N):
    #Calculation of the isotropic part of a signal
    for i in range(1,N):
        if i%2 == 0:
            npData[:,i] -= darkX
        else:
            npData[:,i] -= darkY
    G_factor = np.sum(npData[:,2])/np.sum(npData[:,1])
    npData[:,2] = npData[:,2] / G_factor
    npData[:,4] = npData[:,4] / G_factor
    plt.figure(1)
    plt.plot(t, npData[:,1],'.',markersize=2)
    plt.plot(t, npData[:,2],'.',markersize=2)
    plt.figure(2)
    plt.plot(t, npData[:,3],'.',markersize=2)
    plt.plot(t, npData[:,4],'.',markersize=2)
    return (npData[:,3] + 2*npData[:,4])/3


"""******Subtract dark signal from the experimental data and get isotropic part of the signal******"""
darkX = 71
darkY = 65
I_iso = calculate_isotropic_part(darkX,darkY,len(npData[0,:]))
sigma2 = np.maximum(I_iso,np.ones(len(I_iso))) #removing all of the values which are less than one count
"""************************************************************************************************"""


"""++++++++++++++++Fix boundaries of a search area down here++++++++++++++++"""
bounds = [(0,10000),#T (Shift of the model function to the IRF in time domain)  
          (0, 100), #a0(free term in the  model function)
          (0, 7000),#a1
          (0,7000),  #a2
          (0,7000),  #a3
          #(1.,30.), #a4
          (0,7000),#t1
          (0,7000), #t2
          (0,7000)   #t3
          #(1,30.)   #t4
         ]
"""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""


"""---------Define model decay function here--------------"""
def model_decay_function(T,a0,a1,a2,a3,t1,t2,t3):
    #assumed function of the isotropic part of a fluorescence decay
    return a0 + a1*np.exp(-t/t1)+a2*np.exp(-t/t2)+a3*np.exp(-t/t3), int(round(T/dt));
"""-------------------------------------------------------"""



def observable_decay_function(*parameters):
    #calculation of observable in the experiment decay function by means of convolution through fft
    model_function, shiftT = model_decay_function(*parameters)
    fourier_I_model = np.fft.fft(model_function)
    IRF_shifted = np.roll(IRF,shiftT)
    fourier_IRF_shifted = np.fft.fft(IRF_shifted)
    return dt * np.fft.ifft(fourier_I_model * fourier_IRF_shifted).real


"""============ Choose the objective functions to minimize=================="""
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
    result = differential_evolution(least_square_roots, bounds,  mutation = 0.5,maxiter = 10000,  recombination = 0.3, popsize = 10 )
    stop = timeit.default_timer()
    execution_time = stop - start
    print("Program Executed in "+str(execution_time))
    return result

result = finding_the_minimum()
parameters = result.x
print(result.success)
print(result.message)
print(parameters)
plt.figure(3)    
plt.plot(t, I_iso - observable_decay_function(*parameters), '.',markersize=5)
plt.figure(4)
plt.plot(t, I_iso,'.', color = 'r',markersize=2)
plt.plot(t,observable_decay_function(*parameters), linewidth = 0.5, color = 'b')
plt.figure(6)
n, bins, patches = plt.hist(I_iso - observable_decay_function(*parameters), 50, density=True, facecolor='g', alpha=0.75)
plt.show()
