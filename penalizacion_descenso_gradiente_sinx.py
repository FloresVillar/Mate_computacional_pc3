import numpy as np
import math as mt
"""f(x)=sin(x)  sujeto a x^2 < 1"""
def f(x):
    return mt.sin(x)

def g(x):
    return x**2 - 1

def L(x,u):
    return mt.sin(x)+u*(x**2-1)

def dx(x,u):
    return mt.cos(x)+u*2*x

def grad_desc(x,u, lr=0.01, epsilon=0.0001, nMax=10):
    delta = np.inf
    i = 1
    while i < nMax and delta > epsilon:
        gradient = dx(x,u)
        print(f"\t\titer{i}: x = {round(x,5)}, gr = {round(gradient,5)}")
        x1 = x - lr * gradient
        if x1 < -1 or x1 > 1:
            x1 = np.clip(x1, -1, 1)
        delta = mt.fabs(x1 - x)
        x = x1
        if(g(x)==0):
            break
        i += 1
    return x

if __name__ == '__main__':
    x = 0  
    u = 1
    nMax=20
    y = 1.5
    for i in range(1,nMax):
        x = grad_desc(x,u)
        print(f"i:{i}: x={round(x,5)}, g(x)={round(g(x),5)}, u ={round(u,5)}")
        if g(x) == 0:
            break
        u *=y