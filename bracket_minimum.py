#Se asume que son funciones unimodales, solo un minimo
#import matplotlib.pyplot as plot
import numpy as np
from statistics import mean
def f(x):
    return (x-1)**2
#el libro proporciona una buena explicacion
def bracket_minimum(x, nMax, s=10**-2, k=1): #como indica la teoria del libro→
    a, ya  =x, f(x)                             # el exito de un algoritmo es sensible→ 
    b, yb = a + s, f(a + s)                     # →a la eleccion del hiperparametro →
    if yb > ya: # si avanzara de der a izq (←)  # →en este caso para k=1 se cumple en 100 iteraciones→
        a, b = b, a                             # → con el minimo en x= 1 f(1) = 0 lo cual es exacto
        ya, yb = yb, ya                         # → pero con k=2 x=1.6 f(1.6) = 0.36 resultado pobre→
        s = -s                                  # → con k=0.5 no se cumple nunca obviamente... →
    i = 0                                       # → con k = 5 da cualquier cosa
    while i < nMax:
        c, yc = b + s, f(b +s)
        if yc > yb:     # ni bien se cumple termina retorna
            if a < c:
                return  i+1, np.array([a,c]) 
            else: 
                return  i+1, np.array([c,a])
        a, ya, b, yb = b, yb, c, yc 
        s = s * k
        i = i + 1
    if i == nMax:  #el retorno si no se cumple el if interno
        return i,'con esta cantidad de iteraciones no se consigue'
    
#se puede inferir que avanzara muy lentamente, 
#el tamaño del paso es 0.01
def grafica():
    X = np.linspace(0,2,200)
    Y = (X-1)**2

if __name__=='__main__':
   x0 = 0
   n = 500
   i, I = bracket_minimum(x0, n)    #notar que siempre espera un retorno, si en while no se cumple entonces habria un error None unupset no se podria descomprimir
   print('iteracioes i:'+str(i)+'  intervalo:'+str(I))
   if type(I) is not str:
        print('min:'+str(round(mean(I),5))+'\n'+'f(min):'+str(round(f(mean(I)),5)))