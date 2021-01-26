#imports module for numerical calculations
import numpy as np
#import module for scientific calculations
import scipy.linalg as la
#imports module for visualization of results
import matplotlib.pyplot as plt

def D(E,W):
    X = E + W
    return  4*E*X/(4*E*X + W**2*(np.sin(np.sqrt(C*X)))**2)

def number_of_levels():
    return int(np.sqrt(C*U)/np.pi + 1)
C = 5
U = 20
N = number_of_levels()
print(N)
U1 = np.pi**2*N**2/C
dU = U1 - U
E = np.linspace(0, 50, 1000)

plt.figure(1)
plt.title("Initial Quantum Well")
plt.plot(E, D(E,U), color ='r')
plt.plot(dU,0,'x')
plt.xlabel('Energy, E')
plt.ylabel('Transmission coefficient, T(E)')

plt.figure(2)
plt.title("Deeper Quantum Well")
plt.plot(E, D(E,U1), color ='r')
plt.plot(0,0,'x')
plt.xlabel('Energy, E')
plt.ylabel('Transmission coefficient, T(E)')

plt.show()
