#imports module for numerical calculations
import numpy as np
#import module for scientific calculations
import scipy as sc
import scipy.linalg as scla
#imports module for visualization of results
import matplotlib.pyplot as plt
import cmath
from scipy import constants

def analytic_transmission():
    N = 1000
    E = np.linspace(0.,Emax,N)
    V = np.abs(E-U)
    E1 = 4*E*V
    E2 = C * np.sqrt(2*m1*a**2*V)
    T = np.empty_like(E)
    for i in range(N):
        if  U > E[i]:
            T[i] = E1[i]/(E1[i]+(U*np.sinh(E2[i]))**2)
        else:
            T[i] = E1[i]/(E1[i]+(U*np.sin(E2[i]))**2)
    plt.figure(1)
    plt.plot(E, T,linestyle = '-', color = 'r')
    plt.legend("T(E)")
    plt.xlabel("E, eV")
    plt.ylabel("T")
    plt.title("Аналитическое T(E)")
    
def numerical_solve():
    #число точек по оси абсцисс
    N1 = 200
    #число узлов полиномов Лежандра
    N = 50
    E = np.linspace(0.,Emax,N1)
    T = np.zeros(len(E))
    for k in range(1, len(E)):
        psi = solve_equation(N,E[k])
        T[k] = np.abs(psi[N - 1]) ** 2
    plt.figure(2)
    plt.plot(E, T ,linestyle = '-', color = 'b')
    plt.xlabel("E, eV")
    plt.ylabel("T")
    plt.title("Численное T(E)")
    plt.show()
def solve_equation(N,E):
    points, weights = np.polynomial.legendre.leggauss(N)
    k = C * np.sqrt(2*m1*E)
    G = np.zeros((N, N), dtype=complex)
    F = np.zeros(N, dtype=complex)
    for j in range(N):
        F[j] = np.exp(1j*k*a/2*points[j])
        for l in range (N):
            G[j,l] = -C**2*1j*m1*U*a/(2*k)*np.exp(1j*k*a/2*np.abs(points[j] - points[l]))*weights[l]
    G1 = np.eye(N, dtype=complex) - G
    return  np.linalg.solve(G1,F)

    
#мировые константы
m = constants.m_e
h = constants.hbar
eV = 1.60218e-19
A = 1.e-10
#константа перехода между системами едениц
C = A/h*np.sqrt(eV*m)
#ширина ямы в ангстремах
a = 15
#масса в долях массы электрона
m1 =1
#высота барьера в эВ
U = 0.25
#максимальная энергия в Эв
Emax = 4.
analytic_transmission()
numerical_solve()
