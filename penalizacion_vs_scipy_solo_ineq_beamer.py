import numpy as np 
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#f(x,y)=x^2 + xy + y^2 -2y 
#sujeta a   x + y-2 < 0
#F(x,y) = x^2 +xy +y^2 -2y +mu*(x+y-2)^2 
def f(x):
    return x[0]**2 +x[0]*x[1] +x[1]**2 -2*x[1] 

def P(x):
    return (x[0]+x[1]-2)**2

def g(x):
    return -(x[0]+x[1]-2)
#F(x,y) = x^2 +xy +y^2 -2y  
def G1(x,mu):
    dx = 2*x[0]+x[1]
    dy = 2*x[1] +x[0]
    return np.array([dx,dy])
#F(x,y) = x^2 +xy +y^2 -2y +mu*(x+y-2)^2 
def G2(x,mu):
    dx = 2*x[0]+x[1]+2*mu*(x[0]+x[1]-2)
    dy = 2*x[1] +x[0]-2+2*mu*(x[0]+x[1]-2)
    return np.array([dx,dy])

def grad_desc_for_penalizacion(x:np.array,mu,G, lr=0.1,nMax=100,epsilon=0.0001):
    i = 0
    gr = G(x,mu)  
    while i<nMax and np.linalg.norm(gr)>=epsilon:
        x = x -lr*gr
        gr = G(x,mu)
        print(f"\tx = {[round(float(val),5) for val in x]}, gr = {[round(float(val),5) for val in gr]}")
        i = i + 1
    return x

def penalizacion(x,mu,epsilon,beta,nMax):
    i = 1
    while i < nMax :
        print(f"g(x)={round(g(x),5)}")
        if max(0,g(x))==0:
            x = grad_desc_for_penalizacion(x,mu,G1)
        else:
            x = grad_desc_for_penalizacion(x,mu,G2)
        print(f"Iteración {i}: x = {[round(float(val),5) for val in x]}, P(x) = {round(P(x),5)},u={mu}")
        if mu * P(x) < epsilon:
            return x
        else:
            mu = mu * beta
            i = i + 1
    print("maximo numero de iteracciones")

def scipy_test():
    restriccion = dict(type = 'ineq',fun = g)
    x_opt = optimize.minimize(f, (0, 0), method='BFGS').x
    x_cons_opt = optimize.minimize(f, (0, 0), method='SLSQP',constraints=restriccion).x
    return x_cons_opt


if __name__=='__main__':
    x1 = np.array([3,4])
    mu = 0.5
    beta = 0.5
    epsilon = 0.001
    nMax = 50
    x=penalizacion(x1,mu,epsilon,beta,nMax)
    #x = grad_desc(x1)
    print("el minimo en :")
    print(x)
    print("lo que da scipy")
    print(scipy_test())

 