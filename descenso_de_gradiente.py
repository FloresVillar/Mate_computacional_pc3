import numpy as np
from scipy.optimize import minimize
def f(x):
    return (x[0]**2+x[1]**2+x[0]*x[1]-2*x[1])

def ff(v):
    x,y = v
    return (x**2+y**2+x*y-2*y)

def gf(x):
    return np.array([2*x[0]+x[1],x[0]+2*x[1]-2])

def grad_desc(x:np.array,lr=0.1,nMax=200,epsilon=0.0):
    points = []
    i = 0
    gr = gf(x)
    while i<nMax and np.linalg.norm(gr)>=epsilon:
        x = x -lr*gr
        gr = gf(x)
        points.append(x)
        i = i + 1
    return x

if __name__=='__main__':
    x1 = np.array([3,4])
    x = grad_desc(x1)
    print(x)
    print("scipy")
    x = minimize(ff,x0=[3,4])
    print(x.x)
"""
    comparando descenso de gradiente vs scipy 
    x = grad_desc(x1)
    print("el minimo en :")
    print(x)
    print("comprobando con scipy")
    x = minimize(ff,x0=[3,4])
    print(x.x)"""