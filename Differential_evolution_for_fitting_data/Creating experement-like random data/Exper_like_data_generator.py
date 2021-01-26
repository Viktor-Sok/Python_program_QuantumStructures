import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read in an Instrumental Respond Function(IRF)
IRF_data = pd.read_table('IRFy_wm_free.dat',\
    header = None)
npIRF = IRF_data.to_numpy()
t = npIRF[:,0]
dt = t[1]-t[0]
IRF = npIRF[:,1]/(np.sum(npIRF[:,1])*dt) # normalizing IRF function
print(dt)

def export(x,data,f, sigma):
    out = pd.DataFrame(np.array([x,data,f]).transpose())
    file_name = 'Random_Data_experiment'+ '_Sigma_'+ str(sigma) 
    out.to_csv(file_name + '.dat',sep = '\t' , header = None, index = None)

    
def fun(x):
    # exponential isotropic decay function
    return 15.+ 310*np.exp(-x/650) + 380*np.exp(-x/130) + 160*np.exp(-x/3750)

def observable_decay_function(T):
    #calculation of observable in the experiment decay function by means of convolution through fft
    shiftT = int(round(T/dt))
    fourier_I_model = np.fft.fft(fun(t))
    IRF_shifted = np.roll(IRF,shiftT)
    fourier_IRF_shifted = np.fft.fft(IRF_shifted)
    return dt * np.fft.ifft(fourier_I_model * fourier_IRF_shifted).real

f = observable_decay_function(2400)
plt.figure(1)
plt.plot(t,f)
mu, sigma = 0, 11.7
rand = np.random.normal(mu, sigma, len(t))
#data = f - rand + 30*np.random.rand(len(t)) #gau
#data = f - rand + np.random.poisson(10, len(t)) - 10
data = f - rand
export(t, data, f, sigma)
                 
plt.figure(2)
plt.plot(t, data)
plt.figure(3)
n, bins, patches = plt.hist(f - data, 50, density=True, facecolor='g', alpha=0.75)                 
plt.show()
