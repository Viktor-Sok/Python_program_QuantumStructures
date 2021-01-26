import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal

def convolution(y,d):
    
    fy1 = np.fft.fft(np.exp(-y))
    K = d * np.fft.ifft(fy1*fy1).real
    return K

def test_fun_1(y):
    f = y * np.exp(-y)
    plt.figure(1)
    plt.plot(y, f)
    return f
def convolution_scipy(y):
    f1 = np.exp(-y)
    K = signal.fftconvolve(f1, f1, mode = 'same')
    return K

y = np.linspace(0.,8.,3000)
d = y[1] - y[0]
f1 = test_fun_1(y)
ff1 = convolution(y,d)
plt.figure(2)
plt.plot(y, ff1)
d1 = ff1 - f1
plt.figure(3)
plt.plot(y, d1)
fff1 = convolution_scipy(y)
plt.figure(4)
plt.plot(y, fff1)
plt.show()
