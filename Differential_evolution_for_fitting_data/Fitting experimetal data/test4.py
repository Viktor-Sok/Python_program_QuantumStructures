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
t = npData[:,0]*1000
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

I_iso = calculate_isotropic_part(71,65,len(npData[0,:]))
for i in range(len(I_iso)):
    if I_iso[i] <= 1.:
        print(I_iso[i])
print(len(npData[0,:]))
