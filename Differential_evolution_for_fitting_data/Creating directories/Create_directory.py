import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def createMainFolder(directory):
    os.makedirs(directory, exist_ok = True)

createMainFolder('./exper1/1_2/')
os.chdir("./exper1/")
path1 = os.getcwd()
os.chdir(path1 + "/1_2/")
path2 = os.getcwd()

#parent_dir = "./exper1/"

##def createSubFolder (directory):
##    #path = os.path.join(parent_dir, directory)
##    os.makedirs(directory,exist_ok = True )

##print(path1)

#createMainFolder(parent_dir + "1_2/")

#path = os.getcwd()
def new_name(name,path, newseparator='_'):
    #name can be either a file or directory name
    os.chdir(path)
    base, extension = os.path.splitext(name)
    i = 1
    while os.path.exists(name):
        name = base + newseparator +'('+ str(i)+')' + extension
        i += 1

    return name

x= np.linspace(0, 10, 100)
plt.plot(x, x**2)
plt.savefig(new_name("graph.png", path2))

A = np.array([1,2,3])
B = np.array([4,5,6])
C1 = np.array([A,B])
header =["num1", "num2"]
index = ["one1","two2","three3"]
data = pd.DataFrame(C1,index = header,columns = index ).transpose()
file_name ='Test' +  '.dat'
data.to_csv(new_name (file_name, path1) ,sep = '\t' )

