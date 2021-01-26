from sympy import Matrix,eye, zeros, ones, diag, GramSchmidt,pprint
from sympy import(
    symbols,   # define symbols for symbolic math
    diff,      # differentiate expressions
    integrate, # integrate expressions
    Rational,  # define rational numbers
    lambdify,  # turn symbolic expr. into Python functions
    )
import sympy as sym
from sympy import expand, factor, simplify, sqrt, exp, I, Transpose

p, p1, kx, ky, kz, Q, kp, km    = symbols('p p1 kx ky kz Q kp km')
a = sqrt(Rational(1,2))
b = sqrt(Rational(2,3))
c = sqrt(Rational(1,6))
P1 = Matrix(2,6,[p1*kx, p1*ky, p1*kz, 0, 0,0, 0,0,0,p1*kx, p1*ky, p1*kz])
P2 = Matrix(6,4,[a*kz*Q, b*I*ky*Q, c*kz*Q, 0,
                 -I*a*kz*Q, I*b*kx*Q, I*c*kz*Q, 0,
                 a*km*Q, 0, c*kp*Q, 0,
                 0, c*kz*Q, I*b*ky*Q, a*kz*Q,
                 0, -I*c*kz*Q, I*b*kx*Q, I*a*kz*Q,
                 0, c*km*Q, 0, a*kp*Q
    ])
P3 = p*Matrix(4,2,[-a*km, 0, b*kz, -c*km, c*kp, b*kz, 0, a*kp
    ])


P11 = P3.T.subs([(km,p1),(kp,km)])
P11 = P11.subs(p1,kp)
P22 = Matrix(4,6,[a*kz*Q, I*a*kz*Q, a*kp*Q, 0, 0, 0,
                  -b*I*ky*Q,-I*b*kx*Q, 0,c*kz*Q, I*c*kz*Q, c*kp*Q,
                  c*kz*Q, -I*c*kz*Q,c*km*Q, -I*b*ky*Q, -I*b*kx*Q, 0,
                  0, 0, 0, a*kz*Q, -I*a*kz*Q, a*km*Q
    ])
P33 = P1.T

P = P1*P2*P3 + P11*P22*P33
print(P.shape)
pprint(simplify(P))
P = P.subs([(kp, kx + I*ky),(km, kx - I*ky )])
pprint(simplify(P))
expr = kx**2*(ky**2-kz**2)**2+ky**2*(kz**2-kx**2)**2+ kz**2*(kx**2-ky**2)**2
pprint(expand(expr))
