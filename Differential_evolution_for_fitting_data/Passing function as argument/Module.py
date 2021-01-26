
def input(x,y,fun,funname):
    def lol(x,y):
        print(x/y)
    def name(funname):
        funname(x,y)
    name(funname)   
    return fun(x,y)
