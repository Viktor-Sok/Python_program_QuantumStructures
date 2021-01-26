#imports module for numerical calculations
import numpy as np
#import module for scientific calculations
import scipy as sc
import scipy.linalg as scla
#imports module for visualization of results
import matplotlib.pyplot as plt
from scipy import optimize
def transmisson_coeff(d,kx,ky,H):
    N = len(ky)
    Tx = np.empty_like(ky)
    Rx = np.empty_like(ky)
    for j in range(N):
        E = (kx**2 + ky[j]**2) /2
        q = ky[j] + d/2
        k1 = np.sqrt(2*E - (q-d/2)**2)
        k2 = np.sqrt(2*E -(q+d/2)**2)
        r = solve(E,k1,k2,d,H,q)
        #Tx[j] = k2/k1*np.abs(t)**2
        Rx[j] = np.abs(r)**2
    return Rx

def solve(E,k1,k2,d,H,q):
    #матрица переноса
    M = np.identity(2, dtype = complex)
    x = np.linspace(-d/2, d/2, H)
    h = x[1] - x[0]
    for j in range(H):
        kj = np.sqrt(2*E - (q+x[j])**2)
        Mj = np.array([[np.cos(kj*h),1/kj*np.sin(kj*h)],[-kj*np.sin(kj*h),np.cos(kj*h)]], dtype = complex)
        M = np.dot(Mj,M)
    #коэффициенты r,t
    #A = (M[1,0] - 1j*k1*M[1,1])/(M[0,0] - 1j*k1*M[0,1])
    #t = np.exp(-1j*(k1+k2)*d/2)*(A*(M[0,0]+1j*k1*M[0,1]) - M[1,0] - 1j*k1*M[1,1])/(A - 1j*k2)
    a = 1j*k2*M[0,0] - M[1,0]
    b = k1*k2*M[0,1] + 1j*k1*M[1,1]
    r = np.exp(-1j*k1*d)*(b-a)/(b+a)
    return r

    
     
