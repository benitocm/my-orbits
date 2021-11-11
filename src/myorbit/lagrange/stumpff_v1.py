
from math import cos,sin,cosh,sinh

import functools
from math import factorial
@functools.lru_cache(maxsize=128)
def ck(t,k,N=1,delta=1e-15):
    while abs(t**N/factorial(2*N+k+2))>delta:N+=5
    sk=lambda n:t/((2*n+k+1)*(2*n+k+2))*(1-sk(n+1)) if n<N else 0
    return (1-sk(0))/factorial(k)

@functools.lru_cache(maxsize=128)
def ck_noerror(t,k,N=20):
    sk=lambda n:t/((2*n+k+1)*(2*n+k+2))*(1-sk(n+1)) if n<N else 0
    return (1-sk(0))/factorial(k)

def error_ck(t,k,N):
    return abs(t**N/factorial(2*N+k+2))

def ck_analytic(t):
    if t==0:
        c0=1
        c1=1
        c2=1/2
        c3=1/6
    else:
        y=abs(t)**0.5
        c0=cos(y) if t>=0 else cosh(y)
        c1=sin(y)/y if t>=0 else sinh(y)/y
        c2=(1-cos(y))/y**2 if t>=0 else -(1-cosh(y))/y**2
        c3=(y-sin(y))/y**3 if t>=0 else -(y-sinh(y))/y**3
    return c0,c1,c2,c3

def stumpff_C(z):
    #_, _, c2, _ = ck_analytic(z)
    #return c2
    return ck(z,2)

def stumpff_S(z):
    #_, _, _, c3 = ck_analytic(z)
    return ck(z,3)

