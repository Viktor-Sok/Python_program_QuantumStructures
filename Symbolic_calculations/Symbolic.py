from sympy import Matrix,eye, zeros, ones, diag, GramSchmidt,pprint
from sympy import(
    symbols,   # define symbols for symbolic math
    diff,      # differentiate expressions
    integrate, # integrate expressions
    Rational,  # define rational numbers
    lambdify,  # turn symbolic expr. into Python functions
    )
import sympy as sym
from sympy import expand, factor, simplify
#from sympy import init_printing

# Manual here: https://docs.sympy.org/latest/tutorial/simplification.html

#from sympy import* - импорт всего из sympy

#задание символов
a, b, c = symbols('a b c')
# создание символьного выражения
A = Matrix(2,3,[1,0,1,a,b,0])
pprint(A)
B = Matrix(3,2,[1,c,0,b,0,1])
pprint(B)
#умножение матриц 
C =  A*B
pprint(C)
#создание из символьного выражения питоновской функции (вместо numpy можно и другую блиблиотеку использовать math например)
C_matrix = lambdify([a, b, c], C,"numpy")
#подстановка определённых значений переменных
C1 = C_matrix(1,2,3)
pprint(C1)
# другой способ подстановки вместо а - 1
pprint(C.subs([(a,1),(b,2),(c,3)]))
#можно вставлять выражения друг в друга
expr = a**2
G2 = expr.subs(a,b+c)
pprint(G2)
#операции рациональные с корнем
x = Rational(3)
y = Rational(2)
pprint(x/y)
z = sym.sqrt(3)
X = y/x*z
pprint(X)
# решение линейных уравнений 1 переменной F(a)=0
F = a**2 - 1
roots = sym.solve(F, a)
pprint(roots)
# решение уравнений F(a) = G(a)
F1 = sym.Eq(a+ 1, 4)
roots1 = sym.solve(F1, a)
pprint(roots1)
# упрощение выражений
expr = (a+b)*(a-b)
pprint(expand(expr))
expr1 = a**2 - b**2
pprint(factor(expr1))
expr2 = (a+b)**2 - a**2 - a*b
pprint(expand(simplify(expr2)))
# проверка равенства двух символьных выражений (численно в случайных точках)
ex1 = sym.cos(a)**2 - sym.sin(a)**2
ex2 = sym.cos(2*a)
print(ex1.equals(ex2))
# Численное вычисления выражения evalf(число знаков)
EXP = a**2 + sym.cos(a)
ANS = EXP.subs(a,4).evalf(4)
print (ANS)


