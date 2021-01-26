"""
Численное вычисление интеграла методом Гаууса с использованием
 корней полиномов Лежандра и Лаггера.
"""
#imports module for numerical calculations
import numpy as np
#import module for scientific calculations
import scipy.linalg as la
#imports module for visualization of results
import matplotlib.pyplot as plt

def error_dependence():
    N_ = 150
    M = [0]*N_
    for i in range(N_):
        M[i] = i+1
    err1 = [0.0]*N_
    err2 = [0.0]*N_
    Int1 = [0.0]*N_
    Int2 = [0.0]*N_
    exact = np.sqrt(np.pi)
    for k in range(N_):
        Int1[k] = gauss_laguerre(k+1)
        Int2[k] = gauss_legendre(k+1)
        err1[k] = abs(exact - Int1[k])
        err2[k] = abs(exact - Int2[k])
    fig1 = plt.gcf()
    plt.plot(M, err1, linestyle = '-',color = 'r')
    plt.plot(M, err2, linestyle = '-',color = 'b')
    plt.title("Ошибка, Лаггер (red), Лежандр(blue)")
    plt.semilogy()
    plt.show()
def gauss_laguerre(N_):
    points, weights = np.polynomial.laguerre.laggauss(N_)
    #задание интегрируемой функции
    f = 2*np.exp(-points**2)
    F = [0.0]*N_
    for i in range(N_):
        F[i] = np.exp(points[i])*f[i]
    return np.dot(weights, F)    
def gauss_legendre(N_):
    points, weights = np.polynomial.legendre.leggauss(N_)
    #задание интегрируемой функции
    f = (4/(1-points)**2)*np.exp(-((1+points)/(1-points))**2)
    return np.dot(weights, f) 

# число корней полиномов
N = 20
int1 = gauss_laguerre(N)
int2 = gauss_legendre(N)
print ('laguerre = ',int1)
print ('legendre = ',int2)
error_dependence()
