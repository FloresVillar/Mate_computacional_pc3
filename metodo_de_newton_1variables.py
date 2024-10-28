import numpy as np
import math as mt 
def f(x):
    return mt.sin(x)

def L(x,u):
    return mt.sin(x)+u*(x**2-1)

def g(x):
    return x**2-1

def dx(x,u):
    return mt.cos(x)+u*2*x

def dxx(x,u):
    return -mt.sin(x)+2*u

def metodo_newton(x,u,epsilon=0.001,nMax=500):
    delta = np.inf
    i = 1
    while i<nMax and delta >epsilon:
        x1 = x - dx(x,u)/dxx(x,u)
        delta =  mt.fabs(x1 -x)
        x  = x1
        print(f"Iteracion{i}:x={x},g(x)={g(x)},u={u}")
        i = i + 1
    return x

if __name__=='__main__':
    x0 = 0
    u = 1
    y=2.5
    for i in range(10):
        print(f"\niteracion {i +1}-")
        x1 = metodo_newton(x0,u)
        print("critico")
        print(x1)
        print('g(x)=0?' +str(g(x1)))
        x0 = x1
        u = min(u*y,10)
""" x1 = metodo_newton(x0,u)
    print("el critico")
    print(x1)
    print('g(x)=0?'+str(g(x1)))
    u *=y
    x2 = metodo_newton(x1,u)
    print("el critico")
    print(x2)
    print('g(x)=0?'+str(g(x2)))
    u *=y
    x3 = metodo_newton(x2,u)
    print("el critico")
    print(x3)
    print('g(x)=0?'+str(g(x3)))
    u*=y
    x4 = metodo_newton(x3,u)
    print("el critico")
    print(x4)
    print('g(x)=0?'+str(g(x4)))
    """