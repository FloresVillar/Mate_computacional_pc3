import numpy as np 
from scipy import optimize  
import math as mt
""" f (x) = -(x-4)^2 + 4
        x <=5       x-5 <=0
        x >=0       0-x <=0
        """
#F(x,mu) = f(x) -1/mu*g1(x) -1/mu*g2(x)
#         =-(x-4)^2 +4 -1/mu*(x-5) -1/mu*(0-x)
def f(x):
    return -(x -4)**2 +4 

def g1(x):
    return x- 5

def g2(x):
    return 0- x

def dx(x,mu):
    return -2*(x-4) + 1/(mu*(x-5)**2)-1/(mu*(0-x)**2)

def dxx(x,mu):
    return  -2 -2/(mu*(x-5)**3)-2/(mu*(0-x)**3)

def metodo_newton(x,mu,epsilon=0.01,nMax=100):
    delta = np.inf
    i = 1
    while i<nMax and delta >epsilon:
        x1 = x - dx(x,mu)/dxx(x,mu)
        delta =  mt.fabs(x1 -x)
        x  = x1
        i = i + 1
        print(f"\tx={round(x,5)}, delta={round(delta,5)}")
    return x

def puntos_interiores(x,mu=1,epsilon=0.0001,beta=5,nMax=100):
    delta = np.inf
    i = 1
    while delta > epsilon :
        x1 = metodo_newton(x,mu)
        delta = np.linalg.norm(x1-x)
        x = x1
        mu = mu * beta
        print(f"i={i},x={round(x,5)}, mu={round(mu,5)},beta={round(beta,5)}")
        i = i + 1
    return x  
if __name__=='__main__':
     x0 = 2
     x = puntos_interiores(x0)
     print("minimo")
     print(x)

"""
Metodo de Barrera
en el mismo sentido , se usa para convertir problemas 
con restricciones en problemas sin restricciones
Problema original: 
 min f(x)   sujeto a gi(x)<= 0    i:1→n
 el metodo introduce una funcion barrera que penaliza
(todo lo que implica) las solucione cercanas a la frontera
de las restricciones, pero que solo se define dentro 
conjunto factible  donde gi(x)<=0

Funcion de barrera :
Para cada restriccion gi(x)<=0 se introduce una funcion 
de barrera logaritmica

B(x) =  - SUMA 1/gi(x)
tiene a inf (+) cuando gi(x) → 0- 
esto es que "penaliza" las soluciones cercanas a 
las fronteras de las restricciones

Problema de la barrera :
EL nuevo problema que se debe resolver es 
min phi(x,t) = f(x) +B(x)/t 
donde t es el parametro que controla el peso de la 
barrera 
A medida que t→ inf , la solucion de este problema 
se aproxia a la solucion del problema original
Se comienza con un valor pequeño para t 
y se aumenta en cada iteracion 

Algoritmo basico :
1.Inicializacion: se elige un punto dentro del interior
del conjunto factible y un valor pequeño de t0
2.Resolver el problema no restringido 
    phi(x, tk) = f(x,tk) + B(x)/tk   se obtiene xk
3. Actualizacion: incrementa el valor de tk   tk+1= mu*tk
4.Repetir: continuar hasta que tk sea lo suficientemente 
grande lo que garantiza que xk sea una buena aproximacion
de la solucion original

ejemplo paso a paso:
considerar   min f(x)=x^2  sujeto a 1-x<=0 
paso 1 : 
        g(x) = 1-x   entonces la funcion barrera
        B(x)= -1/g(x) = -1/(1-x)
paso 2: 
        definir el problema de la barrera
        min phi(x,t) =x^2 -1/t(1-x)
        t es el parametro de barrera
paso 3:
        resolver con un valor t 
        t= 1 
        min phi(x,t) = x^2 -1/1-x
        derivar phi
        2x + 1/(1-x)^2 
        igualar a 0
        2x + 1/(1-x)^2  = 0 solucion x
        este x se usa como punto de partida 
        para la siguiente iteracion con el nuevo t
paso 4:
        actualizacion del valor de t
        t = 10
        min phi(x,t) = x^2 -1/10*(1-x)
        resolver este problema

paso 5:
        iterar hasta la convergencia , a medida
        que se incrementa t x, se acerca a la
        solucion optima, continuar hasta tener
        una buena aproximacion 
conclusion:
el metodo de barrera trabaja introduciendo una 
penalizacion(aumento de gradiente) a medida 
que se acerca a los bordes de la region factible , t 
aumenta se refina la solucion dentro de la region
factible y se aproxima a la frontera de las restricciones

"""
 