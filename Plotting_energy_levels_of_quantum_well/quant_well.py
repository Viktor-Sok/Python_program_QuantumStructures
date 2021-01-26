#imports module for numerical calculations
import numpy as np
#import module for scientific calculations
import scipy.linalg as la
#imports module for visualization of results
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import constants
import math



def f(k):
    "четные уровни энергии"
    return (np.sqrt(k0**2-k**2)*np.cos(k*a) -  k*np.sin(k*a))
def g(k):
    "нечетные уровни энергии"
    return (np.sqrt(k0**2-k**2)*np.sin(k*a) +  k*np.cos(k*a))
def q(k):
    "волновой вектор в барьерах"
    return k*np.sqrt(k0**2/(k**2) - 1)

def analytic():
    k_even = np.array([])
    k_odd =  np.array([])
    psi_even = np.array([])
    psi_odd = np.array([])
    T = True
    le = 0
    r  = np.pi/(2*a)
    while T :
        'Поиск четных уровней и в.ф'
        if (k0**2 - r**2) < 0:
            break
        k = optimize.brentq(f,le,r)
        k_even = np.append(k_even, k)
        psi_even = np.append(psi_even, np.cos(k*a)*np.exp(q(k)*a)) 
        le = le + np.pi/a
        r = r + np.pi/a
    le = np.pi/(2*a)
    r = np.pi/a
    while T:
        'Поиск нечетных уровней и в.ф.'
        if (k0**2 - r**2) < 0:
            break
        k = optimize.brentq(g,le,r)
        k_odd = np.append(k_odd, k)
        psi_odd = np.append(psi_odd, np.sin(k*a)*np.exp(q(k)*a))
        le = le + np.pi/a
        r = r + np.pi/a
    #нахождение последнего уровня
    even = False
    N = len(k_even) + len(k_odd)
    if N%2 == 0:
        even = True
    le = N*np.pi/(2*a)
    if not even:
        k = optimize.brentq(g,le,k0)
        k_odd = np.append(k_odd, k)
        psi_odd = np.append(psi_odd, np.sin(k*a)*np.exp(q(k)*a))
    else:
        k = optimize.brentq(f,le,k0)
        k_even = np.append(k_even,k)
        psi_even = np.append(psi_even, np.cos(k*a)*np.exp(q(k)*a)) 
    #построение энергий
    K = np.insert(k_even,np.arange(1,len(k_odd)+1,1),k_odd)
    analytic.N1 = len(K)
    E = K**2*h**2/(2*m*eV)
    analytic.E1 = E
    #Построение волновых функций
    Psi = np.insert(psi_even,np.arange(1,len(psi_odd)+1,1),psi_odd)
    plt.figure (1)
    N = 100
    x = np.linspace(-2*a/nm, 2*a/nm, N)
    scale = (E[1]- E[0])/2
    plt.title("Волновые функции из аналитического решения")
    plt.xlabel("координата, нм")
    plt.ylabel("энергия, eV ")
    plt.plot(x, np.where(np.abs(x) < a/nm, 0, V/eV), linestyle='--')
    for j in range (len(Psi)):
        if j%2 == 0 :
            WFj =np.where(np.abs(x) < a/nm, scale*np.cos(K[j] *x*nm),\
                          scale*Psi[j] * np.exp(- q(K[j]) * np.abs(x)*nm))
            graph1, = plt.plot(x, E[j] + WFj,color = 'r', linestyle = '-')
        else:
            WFj =np.where(np.abs(x) < a/nm, scale*np.sin(K[j] *x*nm),\
                          np.sign(x)*scale*Psi[j] * np.exp(- q(K[j]) * np.abs(x)*nm))
            graph2, = plt.plot(x, E[j] + WFj,color = 'b', linestyle = '-')
    plt.legend([graph1, graph2], [ "Чётные","Нечётные"])


    
def numerical_solve_well():
    #Ширина границы зануления в.ф. в нм
    L = 30*nm
    #Ширина ямы в обезразмеренных переменных
    a_ = 2*a/L
    #параметр определяющий близость ямы  с конечными барьерами к яме с бесконечными барьерами
    v=(1-a_)/(2*a_)
    #Число пробных функций  суть N_-2
    N_=1000
    x_ = np.linspace(0.,1.,N_)  
    #Построение потенциала в точках сетки
    #B = (h**2)/(m1*(nm**2)*eV)
    
    v_=[0.0]*N_
    #Высота потенциала в обезразмеренных переменных
    U = 2*V*m*L**2/h**2
    for k in range(N_):
        if x_[k] <= v*a_ :
            v_[k] = U
        else:
            if x_[k]>=v*a_+a_ :
                v_[k] = U
            else:
                v_[k] = 0.0
    
    m_size = len(x_)-2

    M_mat = np.zeros((m_size, m_size))
    I_mat = np.zeros((m_size, m_size))
    
    for k in range(m_size):
        # Note that the numbers are shifted because solution vector 
        # is inside (0,1) while coordinates and potential are in [0,1]
        L_l, L_r = x_[k+1]-x_[k], x_[k+2]-x_[k+1]
        v_l, v_m, v_r = v_[k], v_[k+1], v_[k+2]
        # left part
        M_mat[k,k] += 1./L_l + 1./L_r
        M_mat[k,k] += v_m*(L_l+L_r)/4.
        I_mat[k,k] += (L_l+L_r)/3.
        if k>0:
            M_mat[k,k] += v_l*L_r/12.
            M_mat[k,k-1] += -1./L_l 
            M_mat[k,k-1] += (v_l+v_m)*L_l/12.0
            I_mat[k,k-1] += L_l/6.
        if k<m_size-1:
            M_mat[k,k] += v_r*L_r/12.
            M_mat[k,k+1] += -1./L_r 
            M_mat[k,k+1] += (v_m+v_r)*L_r/12.
            I_mat[k,k+1] += L_r/6.

    #solve eigen problem: return eigen energies (column of size num_eigvals) 
    #and eigen vectors (matrix of size num_trial*num_eigvals)         
    energies, wfs = la.eigh(M_mat, I_mat, eigvals=(0,analytic.N1+3))
    energies = energies * h**2/(2*m*L**2*eV)
    M_ = np.zeros(len(energies))
    #for l in range (len(energies)):
    #    M_[l] = l+1
    print(energies)
    plt.figure(2)
    E = analytic.E1
    graph1, = plt.plot(E,'o', color = 'b')
    graph2, = plt.plot(energies,'x', color = 'r')
    plt.title("Cравнение численного и аналитического решения")
    plt.xlabel("номер уровня")
    plt.ylabel("E, eV ")
    plt.legend([graph1, graph2], [ "Аналитическое","Численное"])
    plt.show()             
    
        
#Мировые константы
me = constants.m_e
h = constants.hbar
# Переводные коэф. системы едениц
eV = 1.60218e-19
nm = 1.e-9
"""
здесь выставлять глубину потенциала
"""
# Глубина потенциала - в эВ
V = 1.5*eV
# Половина ширины ямы в нм
a = 5*nm
# Масса в долях от массы свободного электрона
m = 0.1*me
# Максимальный в.в. связанных состояний
k0 = np.sqrt(V*2*m/h**2)
analytic()
numerical_solve_well()
