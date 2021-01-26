"""
Уравнение Пуассона методом Галёркина в базисе плоских волн.
"""
#imports module for numerical calculations
import numpy as np
#import module for scientific calculations
import scipy.linalg as la
#imports module for visualization of results
import matplotlib.pyplot as plt

def error_dependence(m_,L0_):
    #макс. число пробных функций N
    N = 20
    Err = [0.0]*(N-2)
    X_ = np.linspace(0.,1.,10)
    An_= analytic(L0_,m_,X_)
    Dif = [0.0]*10
    for k in range(3,N+1):
        Num = solve_potential(k,L0_,m_)
        Phi1 = potential_function(X_, Num)
        for l in range(10):
            Dif[l] = abs(Phi1[l] - An_[l])
        Err[k-3] = max(Dif)
    M = [0]*(N-2)
    for i in range(N-2):
        M[i] = i+3
    plt.figure(2)
    plt.title("Зависимость максимальной ошибки от числа пробных функций")
    plt.plot(M, Err,'x', color ='r')
    
    plt.show()
def solve_potential(num_trial, L0_, m_):
    #main matrix: kinetic+potential energy
    T = np.zeros((num_trial, num_trial))
    D = [0.0]*num_trial
    for k in range(1,num_trial+1):
        #diagonal elements
        T[k-1,k-1] = k**2*np.pi**2
        if k == m_:
            D[k-1] = L0_*(1 - np.sin(2*np.pi*k)/(2*np.pi*k))
        else:
            a = np.pi*(k-m_)
            b = np.pi*(k+m_)
            D[k-1] = L0_*(np.sin(a)/a -np.sin(b)/b)
    return np.linalg.solve(T,D)
def potential_function(x, C):
    Phi = [0.0]*len(x)
    for j in range(len(x)):
        for k in range(len(C)):
            Phi[j] = Phi[j] + C[k]*np.sin((k+1)*np.pi*x[j])
    return Phi
def analytic(L0_,m_,X):
    return L0_/((np.pi*m_)**2)*np.sin(np.pi*m*X)
        

# Кэффициент правой части уравнения Пуассона \lambda
L0 = 10.
# Число пробных функций N-2
N = 90
# Построение потенциала с данным параметром m
m = 4.
Coeff = solve_potential(N,L0,m)
plt.figure(1)
X = np.linspace(0.,1.,100)
n = np.sin(np.pi*m*X)
Phi_ = potential_function(X, Coeff)
plt.title("Концентрация (blue), потенциал (red) и аналитическое решение (cyan)  ")
plt.plot(X, Phi_,'-', color ='r')
plt.plot(X, n + L0,'-', color ='b')
An = analytic(L0,m,X)
plt.plot(X, An+L0/2,'-', color ='c')
error_dependence(m,L0)

    
