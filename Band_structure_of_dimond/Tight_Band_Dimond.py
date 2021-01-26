#imports module for numerical calculations
import numpy as np
#import module for scientific calculations
import scipy.linalg as la
#imports module for visualization of results
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import constants
import math
from scipy.sparse import diags

def band_structure_delta():
    # k1 в еденица постоянной решетки a_0 = 5.66А для Германия
    k1 = np.linspace(-np.pi/3, np.pi/3, 100)
    k2 = 0
    k3 = 0
    #массивы точек для разных зон
    e1 = np.empty_like(k1)
    e2 = np.empty_like(k1)
    e3 = np.empty_like(k1)
    e4 = np.empty_like(k1)
    e5 = np.empty_like(k1)
    e6 = np.empty_like(k1)
    for k in range(len(k1)):
        Ek = solve_eigenfunction(k1[k],k2,k3)
        Fk = np.empty_like(Ek)
        # Исключаем вырожденные собственные значения
        for i in range(len(Ek)):
            Fk[i]= round(Ek[i],6)
        Fk,Ik = np.unique(Fk,return_index=True)
        e1[k] = Ek[Ik[0]]
        e2[k] = Ek[Ik[1]]
        e3[k] = Ek[Ik[2]]
        e4[k] = Ek[Ik[3]]
        e5[k] = Ek[Ik[4]]
        e6[k] = Ek[Ik[5]]
    plt.figure(1)
    plt.plot(k1, e1, color ='b')
    plt.plot(k1, e2, color ='b')
    plt.plot(k1, e3, color ='b')
    plt.plot(k1, e4, color ='b')
    plt.plot(k1, e5, color ='b')
    plt.plot(k1, e6, color ='b')
    plt.title("Зонная структура в направлении delta в k - пространстве")
    

    
def band_structure_lambda():
    # k1 в еденица постоянной решетки a_0 = 5.66А для Германия
    k1 = np.linspace(-np.pi/6, np.pi/6, 100)
    k2 = np.linspace(-np.pi/6, np.pi/6, 100)
    k3 = np.linspace(-np.pi/6, np.pi/6, 100)
    #массивы точек для разных зон
    e1 = np.empty_like(k1)
    e2 = np.empty_like(k1)
    e3 = np.empty_like(k1)
    e4 = np.empty_like(k1)
    e5 = np.empty_like(k1)
    e6 = np.empty_like(k1)
    for k in range(len(k1)):
        Ek = solve_eigenfunction(k1[k],k2[k],k3[k])
        Fk = np.empty_like(Ek)
        # Исключаем вырожденные собственные значения
        for i in range(len(Ek)):
            Fk[i]= round(Ek[i],6)
        Fk,Ik = np.unique(Fk,return_index=True)
        e1[k] = Ek[Ik[0]]
        e2[k] = Ek[Ik[1]]
        e3[k] = Ek[Ik[2]]
        e4[k] = Ek[Ik[3]]
        e5[k] = Ek[Ik[4]]
        e6[k] = Ek[Ik[5]]
    #Cтроим от модуля k в направлении lambda
    k = np.sqrt(3)*k1    
    plt.figure(2)
    plt.plot(k, e1, color ='r')
    plt.plot(k, e2, color ='r')
    plt.plot(k, e3, color ='r')
    plt.plot(k, e4, color ='r')
    plt.plot(k, e5, color ='r')
    plt.plot(k, e6, color ='r')
    plt.title("Зонная структура в направлении lambda в k - пространстве")
    
def band_structure_sigma():
    # k1 в еденица постоянной решетки a_0 = 5.66А для Германия
    k1 = np.linspace(-np.pi/6, np.pi/6, 100)
    k2 = np.linspace(-np.pi/6, np.pi/6, 100)
    k3 = 0
    #массивы точек для разных зон
    e1 = np.empty_like(k1)
    e2 = np.empty_like(k1)
    e3 = np.empty_like(k1)
    e4 = np.empty_like(k1)
    e5 = np.empty_like(k1)
    e6 = np.empty_like(k1)
    for k in range(len(k1)):
        Ek = solve_eigenfunction(k1[k],k2[k],k3)
        Fk = np.empty_like(Ek)
        # Исключаем вырожденные собственные значения
        for i in range(len(Ek)):
            Fk[i]= round(Ek[i],6)
        Fk,Ik = np.unique(Fk,return_index=True)
        e1[k] = Ek[Ik[0]]
        e2[k] = Ek[Ik[1]]
        e3[k] = Ek[Ik[2]]
        e4[k] = Ek[Ik[3]]
        e5[k] = Ek[Ik[4]]
        e6[k] = Ek[Ik[5]]
    #Cтроим от модуля k в направлении lambda
    k = np.sqrt(2)*k1    
    plt.figure(3)
    plt.plot(k, e1, color ='g')
    plt.plot(k, e2, color ='g')
    plt.plot(k, e3, color ='g')
    plt.plot(k, e4, color ='g')
    plt.plot(k, e5, color ='g')
    plt.plot(k, e6, color ='g')
    plt.title("Зонная структура в направлении sigma в k - пространстве")
    plt.show()
    
def solve_eigenfunction(k1,k2,k3):
    """
    Решает задачу на собственные значения
    """
    pi = np.pi
    c1 = np.cos(pi/2*k1)
    s1 = np.sin(pi/2*k1)
    c2 = np.cos(pi/2*k2)
    s2 = np.sin(pi/2*k2)
    c3 = np.cos(pi/2*k3)
    s3 = np.sin(pi/2*k3)
    g0 = c1*c2*c3 - 1j*s1*s2*s3
    g1 = - c1*s2*s3 + 1j*s1*c2*c3
    g2 = - s1*c2*s3 + 1j*c1*s2*c3
    g3 = - s1*s2*c3 + 1j*c1*c2*s3
    H = create_matrix(g0,g1,g2,g3)
    E,V = la.eigh(H)
    return E
def create_matrix(g0,g1,g2,g3):
    """
    Задание матрицы гамильтониана
    в модели сильной связи
    """
    H_upper = create_upper_matrix([Es, Vss*g0, 0, 0, 0, Vsp*g1, Vsp*g2, Vsp*g3,
                            Es, -Vsp*np.conj(g1), -Vsp*np.conj(g2), -Vsp*np.conj(g3),
                            0, 0, 0, Ep, 0, 0, Vxx*g0, Vxy*g3, Vxy*g2, Ep, 0 ,
                            Vxy*g3, Vxx*g0, Vxy*g1, Ep, Vxy*g2, Vxy*g1, Vxx*g0,
                            Ep, 0, 0, Ep, 0, Ep     ], 8)
    return symmetrize(H_upper)
def create_upper_matrix(values, size):
    """
    Создаёт верхнюю треугольную матрицу по данным значениеям слева напрво и сверзу вниз
    """
    upper = np.zeros((size, size),dtype=complex)
    upper[np.triu_indices(size, 0)] = values
    return(upper)
def symmetrize(a):
    """
    Cоздаёт эрмитову матрицу по верхней треугольной
    """
    return a + np.conjugate(a.T) - np.diag(a.diagonal())

#Параметры задачи в еV
Es = -5.8
Ep = -5.8+8.41
Vss = - 6.78
Vsp = 5.31
Vxx = 2.62
Vxy = 6.82
band_structure_delta()
band_structure_lambda()
band_structure_sigma()


    
