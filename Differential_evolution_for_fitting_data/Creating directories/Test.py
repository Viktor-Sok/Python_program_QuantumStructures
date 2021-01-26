import os
import matplotlib.pyplot as plt
import numpy as np

print (os.getcwd())
os.chdir("./exper1/1_2")
print(os.getcwd())
print(os.path.exists("graph1.png"))
def new_name(name, newseparator='_'):
    #name can be either a file or directory name
    base, extension = os.path.splitext(name)
    i = 1
    while os.path.exists(name):
        name = base + newseparator + str(i) + extension
        i += 1

    return name

print(new_name("graph.png"))
