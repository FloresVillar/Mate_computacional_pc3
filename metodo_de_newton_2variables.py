import numpy as np
"""f (x , y) = x^2 +xy +y^2 - 2y
        x + y - 2 <=0"""
def Gr(X):
    x,y = X
    dx = 2*x + y 
    dy = x +2*y -2
    return np.array([dx,dy])

def Hessian(X):
    x,y = X
    dx = 2*x + y 
    dy = x +2*y -2
    dxx = 2
    dyx = 1
    dxy = 1
    dyy = 2
    return np.array([[dxx,dyx],[dxy,dyy]])

def metodo_newton(X,nMax=50,epsilon=0.01):
    delta = np.inf
    i = 1
    while delta > epsilon and i<nMax:
        H = Hessian(X)
        gr = Gr(X)
        H_i = np.linalg.inv(H)
        X1 = X-H_i@gr
        delta = np.linalg.norm(X1-X)
        i = i + 1
        X = X1
    return X

if __name__=='__main__':
    X = np.array([1,1])
    X = metodo_newton(X)
    print("minimo")
    print(X)