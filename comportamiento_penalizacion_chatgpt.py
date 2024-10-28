import numpy as np
import matplotlib.pyplot as plt
import math as mt
# Definimos la función original y la penalización
def f(x):
    return np.sin(x)

def p(x):
    return x**2 - 1

# Valores de x
x = np.linspace(-2, 2, 400)

# Valores de u para comparar
u_values = [1, 2, 5]

# Crear la figura
plt.figure(figsize=(10, 6))

# Graficar L(x, u) para diferentes valores de u
for u in u_values:
    L = f(x) + u * p(x)  # Función penalizada
    plt.plot(x, L, label=f'L(x, u) con u={u}')
y = f(x)
plt.plot(x,y,label=f'f(x)=sin(x)')
G = p(x)
plt.plot(x,G,label=f'g(x)=x^2-1')
# Configuraciones de la gráfica
plt.title('Funciones Penalizadas L(x, u) para diferentes valores de u')
plt.axhline(0, color='black', lw=0.5, ls='--')  # Línea y=0
plt.axvline(0, color='black', lw=0.5, ls='--')  # Línea x=0
plt.ylim(-10, 5)  # Ajustar límites del eje y
plt.xlim(-2, 2)   # Ajustar límites del eje x
plt.xlabel('x')
plt.ylabel('L(x, u)')
plt.legend()
plt.grid()
plt.show()
