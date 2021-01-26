#!/usr/bin/python
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.polynomial.laguerre import lagval
from scipy import optimize

def solve_disp_eqn(xs,ws,lmbda):
    """Construct integral equation and find where it has a solution
    """
    eqtn = np.zeros((len(xs),len(xs)))
    # Compute the kernel of integral equation
    def IE_kernel(k_,kk_,lambda_,p_):
        """Kernel of integral equation is integral itself
        """
        res = 1./np.pi/(k_**2+kk_**2+p_**2)
        def kern_fun(kkk_):
            """Function to integrate
            """
            ans = 1./(k_**2+(k_-kkk_)**2+p_**2)/(kk_**2+(kk_-kkk_)**2+p_**2)
            sqrt_ = np.sqrt(2.*p_**2+kkk_**2)
            ans *= sqrt_/(sqrt_+lambda_)
            return ans
        ks = (1.+xs)/(1.-xs)
        dkdx = 2./(1.-xs)**2
        integr = sum(ws*kern_fun( ks )* dkdx) + sum(ws*kern_fun( -ks )* dkdx)
        res -= 2.*lambda_/np.pi**2 * integr
        return res
    # Define the matrix of integral equation
    def det_eqtn(p):
        def phi(k_):
            return 1.-1./(k_*k_+p*p)**0.5
        for i in range(len(xs)):
            ki = (1.+xs[i])/(1.-xs[i])
            for j in range(len(xs)):
                kj = (1.+xs[j])/(1.-xs[j])
                dkdx = 2./(1.-xs[j])**2
                eqtn[i,j] = ws[j]*IE_kernel(ki,kj,lmbda,p)*dkdx
                eqtn[i,j] += ws[j]*IE_kernel(ki,-kj,lmbda,p)*dkdx
            eqtn[i,i] -= phi(ki)
        return la.det(eqtn)
        
    return optimize.brenth(det_eqtn,1.,np.sqrt(2))

    """
    Решение Диракиума методом Хартри - Фока
    """
def solve_HF(lambd):
    
    #нулевое приближение концентрации
    n = np.exp(-2*abs(x))
    Lag = np.zeros((G,L))
    for i in range(L):
        Lag[:,i] = lagval(x, np.eye(1, i+1,i)[0])
    i = 0
    #итерационная процедура
    while i<=M:
        T = matrix_T()
        V = matrix_V(lambd,n,Lag)
        H = T+V
        E1,C1 = np.linalg.eigh(H)
        C = C1[:,0]
        e = E1[0]
        F = find_F(C,x,Lag)
        F = F/np.sqrt(intg(F**2))
        n = F**2*ksi+n*(1-ksi)
        i = i+1
    J = lambd*intg(F**4)
    E = 2*e-J
    return E

def find_F(K,x,Lag):
    "вычисление в.ф"
    F = np.zeros(G)
    for i in range(L):
        F = F+K[i]*Lag[:,i]*np.exp(-x/2)
    return F
        
def matrix_T():
    "матрица кинетической энергии"
    T = np.zeros((L,L))
    for k in range(L):
        for l in range(L):
            if k<l:
                T[k,l] = 0.5*(0.5 + k)
            if k>l:
                T[k,l] = 0.5*(0.5 + l)
            if k==l:
                T[k,l] = 0.5*(0.25 + k )        
    return T

def matrix_V(lambd,n,Lag):
    "матрица потенциальной энергии"
    V = np.zeros((L,L))
    A = np.multiply(w,n)
    for i in range(L):
        for j in range(L):
            B = np.multiply(Lag[:,i],Lag[:,j])
            V[i,j] = -1/2+lambd*np.dot(A,B)
            
    return V

def intg(F):
    "взятие интеграла"
    I=0
    for i in range(L):
        I=I+w[i]*np.exp(x[i])*F[i]
    return 2*I


"""
Задача 9.1
"""
# Диапазон lambda
N = 15
lambd = np.linspace(0.1,2.6,N)
# Число функций Лагера
L = 10
# Число итераций для  нелинейной задачи на собственные значения
M = 100
# Демпфер (для сходимости итерационной процедуры)
ksi = 0.25
# Число корней полиномов Гаусса-Лаггера для взятия интегралов от концентрации
G = 30
x, w = np.polynomial.laguerre.laggauss(G)
E = np.empty_like(lambd)       
for k in range(N):
    E[k] = solve_HF(lambd[k])
plt.figure(1)
plt.plot(lambd, E ,'x', color = 'r')
plt.xlabel("lambda")
plt.ylabel("E")
lambd1 = np.linspace(0.01,2.6,30)
E1 = -1 + 0.5*lambd1 - lambd1**2/12
plt.plot(lambd1,E1, color = 'b')
plt.title(" Хартри-Фок (red), аналитическое(blue)")

"""
Задача 9.2
"""
polyorder = 10
X, W = np.polynomial.legendre.leggauss(polyorder)
#Дмапазон именения lambda
ll = np.linspace(0.1,2.6,N)
E1 = np.empty_like(ll)
for k in range(N):
    E1[k] = -1/2*solve_disp_eqn(X,W,ll[k])**2
plt.figure(2)
plt.plot(ll, E1 ,'o', color = 'b')
plt.plot(lambd, E ,'x', color = 'r')
plt.xlabel("lambda")
plt.ylabel("E")
plt.title("Численное из семинара 8(blue), Хартри-Фок(red)")
plt.show()



