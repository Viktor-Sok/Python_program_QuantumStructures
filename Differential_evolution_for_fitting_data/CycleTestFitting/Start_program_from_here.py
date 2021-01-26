import numpy as np
import Module_of_implementations
import pandas as pd

"""======================================================================="""
"""-= Load the data and insert all of the required parameters down here =-"""
"""======================================================================="""

dataFileName = '2020-11-05 PBS ADH_2mg_ml NADH_40uM 720nm  250mW Sh9 ZET436_10 2000s ZYLZYZYZ_exp.dat'
colm = [[1,2,3,4]] # type all of the  exposition you want to calculate
fileNameIRF = 'IRFy_wm_free.dat'
darkX = 71
darkY = 65
mut = 0.5
recomb = 0.3
pops = 10
maxiter = 10000

"""=======================================++"""
"""== Two, Three and Four exponent models =="""
"""========================================="""
"""

    def model_decay_function(T,a0,a1,a2,t1,t2):
        #assumed function of the isotropic part of a fluorescence decay
        return a0 + a1*np.exp(-t/t1)+a2*np.exp(-t/t2), int(round(T/dt))

    def model_decay_function(T,a0,a1,a2,a3,t1,t2,t3):
        #assumed function of the isotropic part of a fluorescence decay
        return a0 + a1*np.exp(-t/t1)+a2*np.exp(-t/t2)+a3*np.exp(-t/t3), int(round(T/dt))

    def model_decay_function(T,a0,a1,a2,a3,a4,t1,t2,t3,t4):
        #assumed function of the isotropic part of a fluorescence decay
        return a0 + a1*np.exp(-t/t1)+a2*np.exp(-t/t2)+a3*np.exp(-t/t3)+a4*np.exp(-t/t4), int(round(T/dt))    
"""

"""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
"""++++++++++++++++Fix boundaries of a search area down here++++++++++++++++"""
bounds = [(0,10000),#T (Shift of the model function to the IRF in time domain)  
          (0, 100), #a0(free term in the  model function)
          (0, 7000),#a1
          (0,7000),  #a2
          #(0,7000),  #a3
          #(1.,30.), #a4
          (0,7000),#t1
          (0,7000) #t2
          #(0,7000)   #t3
          #(1,30.)   #t4
         ]
"""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
"""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""


i = 1 #counter in order to keep track of execution
transpose_dat = False # become True at the final exposition in order to transpose output dat file
for exposition in colm:
    if i == len(colm):
        transpose_dat = True
    Module_of_implementations.start_calculations( dataFileName, exposition, fileNameIRF, darkX, darkY,
                              mut, recomb, pops, maxiter, bounds,transpose_dat)
    i=i+1

print("Your calculations have been finished! Go to the program folder to see the results.")
