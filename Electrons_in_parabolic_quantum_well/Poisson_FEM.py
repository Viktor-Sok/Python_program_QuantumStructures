#!/usr/bin/python
"""
Уравнение Пуассона методом FEM.
"""
#imports module for numerical calculations
import numpy as np
#import module for scientific calculations
import scipy.linalg as la
#imports module for visualization of results
import matplotlib.pyplot as plt

def error_dependence(m_,L0_):
    #макс. число пробных функций N_ -2
    N_ = 80
    Err = [0.0]*(N_-5)
    for k in range(3,N_-2):
        X_ = np.linspace(0.,1.,k)
        n_ = np.sin(np.pi*m_*X_)
        Num = solve_poisson(X_,n_,L0_)
        An_= analytic(L0_,m_,X_)
        Dif = [0.0]*(N_-5)
        for l in range(k-2):
            Dif[l] = abs(Num[l] - An_[l+1])
        Err[k-3] = max(Dif)
    M = [0]*(N_-5)
    for i in range(N_-5):
        M[i] = i+3
    plt.figure(2)
    plt.title("Зависимость максимальной ошибки от числа пробных функций")
    plt.plot(M, Err,'x', color ='r')
    plt.semilogy()
    plt.show()
    

def analytic(L0_,m_,X):
    return L0_/((np.pi*m_)**2)*np.sin(np.pi*m*X)
def solve_poisson(X_,n_,L0_):
    m_size = len(X_)-2
    T_mat = np.zeros((m_size, m_size))
    D_vec = [0.0]*m_size
    for k in range(m_size):
        L_l, L_r = X_[k+1]-X_[k], X_[k+2]-X_[k+1]
        D_vec[k] = (L0_/6)*(L_l*(n_[k]+2*n_[k+1])+L_r*(2*n_[k+1]+n_[k+2]))
        T_mat[k,k] += 1./L_l + 1./L_r
        if k>0:
            T_mat[k,k-1] += -1./L_l
        if k<m_size-1:
            T_mat[k,k+1] += -1./L_r
            
    return np.linalg.solve(T_mat,D_vec)

# Кэффициент правой части уравнения Пуассона \lambda
L0 = 5.
# Число пробных функций N-2
N = 1002
X = np.linspace(0.,1.,N)
# Построение потенциала с данным параметром m
m = 4.
n = 1/4 - (X-1/2)**2
Phi = solve_poisson(X,n,L0)
plt.figure(1)
scale = 10.
plt.title("Концентрация (blue), потенциал (red) и аналитическое решение (cyan)  ")
plt.plot(X[1:-1], scale*Phi,'-', color ='r')
plt.plot(X, n + L0,'-', color ='b')
An = analytic(L0,m,X)
plt.plot(X, scale*An+L0/2,'-', color ='c')
error_dependence(m,L0)
