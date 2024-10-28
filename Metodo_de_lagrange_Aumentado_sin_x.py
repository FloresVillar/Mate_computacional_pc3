import numpy as np
import math as mt
from statistics import *
""""            f(x) = sin(x)
    sujeto a:   x**2 < 1
                g(x) = x**2 - 1
Lagrange aumentado : 
L(x,lam,p) = f(x) + lam*g(x) + 0.5*p*MAX(0,g(x))**2
            =sin(x) +lam*(x**2-1) +0.5*p*(x**2-1)**2
"""
def f(x):
    return np.sin(x)

def g(x):
    return x**2-1

def Gr1(x,lam,p):
    return  np.cos(x)+lam*2*x 

def Gr2(x,lam,p):
    return np.cos(x)+lam*2*x + 0.5*p*(2*(x**2-1)*2*x)

def desc_grad(G,x,lam,p,lr=0.0001,epsilon=0.001,nMax=100):
        for i in range(nMax):
            gr = G(x,lam,p)
            if np.linalg.norm(gr)<epsilon:
                break
            x = x - lr*gr 
            lr*=0.99
        return x    

def lagrange_aumentado_ineq(x,lam=1,p=1,y=1.02,nMax=150,epsilon=0.0001):
    for i in range(nMax):
        if max(0,g(x))==0:
            x = desc_grad(Gr1,x,lam,p)
        else:
            x = desc_grad(Gr2,x,lam,p)
        lam += p * g(x)  
        p *=y
        print(f"Iteración {i+1}: x = {round(x,6)}, lam = {round(lam,6)}, p = {round(p,6)},g(x)",{round(float(g(x)),6)})
        if abs(g(x))<epsilon:
            break
    return x

if __name__=='__main__':
    x = lagrange_aumentado_ineq(0)
    print(x)
