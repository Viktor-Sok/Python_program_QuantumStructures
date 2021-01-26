import numpy as np
from scipy import optimize
import scipy as sc
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
"""
Решение системы ОДУ dy/dt = F*y
"""
def exponential_decay(t, y):
    F = np.array([[0,1],[-2,-1]])
    dydt = np.dot(F,y)
    return dydt
y0 = np.array([6.02490374e-06,-3.08687006e-06])
t = np.arange(25., 0, - 0.01)

sol = solve_ivp(exponential_decay, [25., 0.], y0, method='BDF',t_eval = t)
plt.figure(3)
print(sol.y[0])
plt.plot(sol.t,sol.y[0] )

plt.show()    
 


