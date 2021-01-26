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
from mpl_toolkits.mplot3d import Axes3D

#степень двойки(для эффективной работы алгоритма FFT)
k = 10
#число точек дискретизации исходной функции
N = 2**k
#величина "обрезанной" оласти определения функции
l = 10
#массив точек исходной функции обрезанной на отрезок 2*l
w1 = np.linspace(-l,l,N)
w2 = np.linspace(-l,l,N)
d = (w1[1] - w1[0])
w1, w2 = np.meshgrid(w1, w2)
f1 =  2*np.pi*np.exp(-(w1**2)/2)*np.exp(-(w2**2)/2)
# выделение куска функции при переодичсеком продолжении f1 , который лежит правее нуля, т.к. FFT  работает от нуля
#f11 = f1[0:int(N/2)]
#f12 = f1[int(N/2): (N)]
f = np.fft.fftshift(f1)
#F = np.block([f12,f11])
# выполняем IFFT
IDF2 = np.fft.ifft2(f)
#Сдвиг отрицательной частотной части влево
IDF2 = np.fft.fftshift(IDF2)
# переходим к соответствующему математическому обратному образу Фурье
MIDF = N**2*d**2*IDF2/(2*np.pi)**2
#выводим соответствующие "координаты из преобразования" 
x1 = np.fft.fftfreq(N,d)
x2 = np.fft.fftfreq(N,d)
#сдвигаем отрицательные частоты влево
x1 = np.fft.fftshift(x1)
x2 = np.fft.fftshift(x2)
#Переходим к еальным координатам
x1 = 2*np.pi*x1
x2 = 2*np.pi*x2
#print (MDF)
#Исходная функция - оригинал
X1 = np.linspace(x1[500],x1[N-500],500)
X2 = np.linspace(x2[500],x2[N-500],500)
X1, X2 = np.meshgrid(X1, X2)
g = np.exp(-X1**2/2)*np.exp(-X2**2/2)
# Сравнение функции и обратного численного преобразований Фурье
#print (1/(2*d),v[N-1])
#print(MDF[int(N/2-6)])
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

SMIDF = MIDF[500:N-500,500:N-500]
x1 = x1[500:N-500]
x2 = x2[500:N-500]
x1, x2 = np.meshgrid(x1, x2)
ax.plot_surface( x1,x2, SMIDF.real,color = 'r')
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, g, color='b')
plt.show()
