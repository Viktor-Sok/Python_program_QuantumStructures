import numpy as np
from inspect import getfullargspec
import pandas as pd

def sum1 (c,d):
    return c + d

args = getfullargspec(sum1).args + [1,2]
print(args)

A = ['a','b','c']
C = A + ['n_' + j for j in A[1:3]]
print(C)

##stri = 'row'
##A = np.array([1,2])
###save to the excel format
##df1 = pd.DataFrame([A],
##                   index= [stri],
##                   columns = args)
##
##
##df1.to_excel("output.xlsx")


