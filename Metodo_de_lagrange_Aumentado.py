import numpy as np 
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""  minimizar f (x y) = (x - 4)^2 + (y - 4)^2
        sujeto a x + y - 5 = 0                """

def f(X):
    x,y =X
    return (x - 4)**2+(y - 4)**2

def h(X):
    x,y= X
    return x + y - 5

#F = f + uh(x)^2/ 2 -lam*h(x)
#F = (x - 4)**2+(y - 4)**2 + 0.5*mu *(x + y - 5)^2-lam*(x + y - 5)

def gF(x,mu,lam):
    dx = 2*(x[0]-4) + 2*0.5*mu*(x[0]+x[1]-5)-lam
    dy = 2*(x[1]-4) + 2*0.5*mu*(x[0]+x[1]-5)-lam
    return np.array([dx,dy])

def grad_desc_for_lagrange_aumented(x:np.array,mu,lam, lr=0.1,nMax=100,epsilon=0.0001):
    i = 0
    gr = gF(x,mu,lam)  
    while i<nMax and np.linalg.norm(gr)>=epsilon:
        x = x -lr*gr
        gr = gF(x,mu,lam)
        i = i + 1
        print(f"\t x={[round(float(val),5) for val in x]}, gr={[round(float(val),5) for val in gr]}")
    return x

def augmented_lagrange_method(x,nMax,mu,beta):
    lam = 0
    for i in range(1,nMax):
        x = grad_desc_for_lagrange_aumented(x,mu,lam)
        lam = lam - mu*h(x)
        mu = mu*beta
        print(f"Iteración {i}: x = {[round(float(val),5) for val in x]}, lam = {round(lam,5)},u={round(mu,5)}")
    return x

def scipy_test():
    restriccion = dict(type = 'eq',fun = h)
    x_opt = optimize.minimize(f, (0, 0), method='BFGS').x
    x_cons_opt = optimize.minimize(f, (0, 0), method='SLSQP',constraints=restriccion).x
    return x_cons_opt


if __name__=='__main__':
    x1 = np.array([1,1])
    x = augmented_lagrange_method(x1,15,1,1) #depende demasiado de los valores de los paramtros
    print("minimo en ")
    print(x)
    print(scipy_test())
     