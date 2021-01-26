import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

def func(x):
    # here one defines a function which should be randomized
    return 5.+ 20*np.exp(-x/100) + 10*np.exp(-x/50) + 5*np.exp(-x/10.)

def export(x,data,f, sigma):
    out = pd.DataFrame(np.array([x,data,f]).transpose())
    file_name = 'Random_Data'+ '_Sigma_'+ str(sigma) 
    out.to_csv(file_name + '.dat',sep = '\t' , header = None, index = None)
    
    

N = 3000 
x = np.linspace(0.,200,N) 

f = func(x)
plt.figure(1)
plt.plot(x,f)

mu, sigma = 0, 0.2
rand = np.random.normal(mu, sigma, N)
data = f - rand
plt.figure(2)
plt.plot(x, data)
export(x, data, f, sigma)
plt.show()
