"""
============================
Вычисление коэффициента прохождения
на kx,ky 
============================


"""
import pylab
import matplotlib.pyplot as plot
import numpy as np
from numpy import ma
import T_solve

#ширина магнитной полоски
d = 4.5
#разбиение по kx и как следствие по ky
N = 500
#число прямоугольных ступенек для приближения потенциала
H = 50
kx_max = 2
ky_min = -5
kx = np.linspace(0.01, kx_max, N)
ky_max = (kx_max**2-d**2)/(2*d)-1.e-8
h = kx[1] - kx[0]
N1 = (ky_max - ky_min) /h
N1 = int(N1)
ky = np.linspace(ky_min , ky_max, N1)
T_kxky = np.zeros((N1,N))
for j in range(N):
    ky_maxj =(kx[j]**2-d**2)/(2*d)-1.e-8
    N1j = (ky_maxj-ky_min)/h
    N1j = int(N1j)
    
    kyj = np.linspace(ky_min,(kx[j]**2-d**2)/(2*d)-1.e-8,N1j)
    Tj = T_solve.transmisson_coeff(d,kx[j],kyj,H)
    for k in range(N1j):
        T_kxky[k, j] = Tj[k]

pylab.xlim([0.01,kx_max])
pylab.ylim([ky_min,ky_max])

# Provide a title for the contour plot
plot.title('Коэффициент отражения, d= {}'.format(d))

# Set x axis label for the contour plot
plot.xlabel(' Kx')

# Set y axis label for the contour plot
plot.ylabel(' Ky')
# Create contour lines or level curves using matplotlib.pyplot module
contours = plot.contour(kx, ky, T_kxky)
# Display z values on contour lines
plot.clabel(contours, inline=1, fontsize=10, fmt ='%1.2f' )
# Display the contour plot
plot.show()

