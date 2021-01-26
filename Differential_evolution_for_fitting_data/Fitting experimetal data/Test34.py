import numpy as np
from inspect import getfullargspec

def sum1 (a,b):
    return a + b
print(sum1(3,4))
print(getfullargspec(sum1).args)
