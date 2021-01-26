from sympy import Matrix,eye, zeros, ones, diag, GramSchmidt,pprint
from sympy import(
    symbols,   # define symbols for symbolic math
    diff,      # differentiate expressions
    integrate, # integrate expressions
    Rational,  # define rational numbers
    lambdify,  # turn symbolic expr. into Python functions
    )
from sympy import*
import sympy as sym
from sympy import expand, factor, simplify

u, t, E = symbols('u t E')
H = Matrix(6,6,[0,0, -t, -t,0,0,  0, 0, t, t,0,0, -t, t, u, 0, 0, 0,  -t, t, 0, u, 0, 0, \
               0,0,0,0,-t, 0, 0, 0, 0, 0,  0, -t ])

D = (H - eye(6)*E).det()
pprint(D)
I = H.eigenvals()
I1 = H.eigenvects()
pprint(I)
pprint(I1)
