#descenso de gradiente
#penalizacion con la funcion
"""Minimizar
 f (x y) = x^2 +xy +y^2 - 2y
 sujeto a 
 x + y + 2 <= 0
 x + y -5 = 0
 
 """
import numpy as np
from scipy.optimize import minimize
#from typing import Callable, Tuple, list

def f(x):
    return (x[0]**2+x[1]**2+x[0]*x[1]-2*x[1])

def g(x):
    return x[0]+x[1]-2

def h(x):
    return x[0]+x[1]-5


def gF(x,mu):
    dx = 2*x[0]+x[1]+2*mu*(x[0]+x[1]-2)+2*mu*(x[0]+x[1]-5)
    dy = 2*x[1] +x[0]-2+2*mu*(x[0]+x[1]-2)+2*mu*(x[0]+x[1]-5)
    return np.array([dx,dy])

#F(x,y) = x^2 +xy +y^2 -2y +mu*(x+y-2)^2+mu*(x+y-5)^2

def grad_desc_for_penalizacion(x:np.array,mu, lr=0.1,nMax=100,epsilon=0.0001):
    points = []
    i = 0
    gr = gF(x,mu)  
    while i<nMax and np.linalg.norm(gr)>=epsilon:
        x = x -lr*gr
        gr = gF(x,mu)
        points.append(x)
        i = i + 1
    return x

def P(x):
    return (x[0]+x[1]-2)**2 + (x[0]+x[1]-5)**2

def penalizacion(x,mu,epsilon,beta,nMax):
    i = 1
    while i < nMax :
        x = grad_desc_for_penalizacion(x,mu)
        if mu * P(x) < epsilon:
            return x
        else:
            mu = mu * beta
            i = i + 1
    print("maximo numero de iteracciones")

if __name__=='__main__':
    x1 = np.array([3,4])
    mu = 0.5
    beta = 0.5
    epsilon = 0.001
    nMax = 50
    x=penalizacion(x1,mu,epsilon,beta,nMax)
    print(x)
   