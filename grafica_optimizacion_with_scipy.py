from pylab import*
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

def func_X_Y_to_XY(f, X, Y):
    """
    Wrapper for f(X, Y) -> f([X, Y])
    """
    s = np.shape(X)
    return f(np.vstack([X.ravel(), Y.ravel()])).reshape(*s)

# función a minimizar
def f(X):
    x, y = X
    return (x - 1)**2 + (y - 1)**2

# minimizo la función si restricciones
x_opt = optimize.minimize(f, (1, 1), method='BFGS').x

# el mínimo para las restricciones
bnd_x1, bnd_x2 = (2, 3), (0, 2)
x_cons_opt = optimize.minimize(f, np.array([1, 1]), method='L-BFGS-B',
bounds=[bnd_x1, bnd_x2]).x

# graficando la solución
"""fig, ax = plt.subplots(figsize=(10, 8))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]),
bnd_x1[1] - bnd_x1[0], bnd_x2[1] - bnd_x2[0],
facecolor="grey")
ax.add_patch(bound_rect)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
ax.set_title('Optimización con restricciones')
plt.show()
"""
#con restricciones tipo inecuacion
def g(X):
    return X[1] - 1.75 - (X[0] - 0.75)**4

def h(X):
    return X[0]+X[1]-5
# definimos el diccionario con la restricción
restriccion = dict(type='ineq', fun=g)
restriccioneq = dict(type='eq',fun=h)
# resolvemos
x_opt = optimize.minimize(f, (0, 0), method='BFGS').x
x_cons_opt = optimize.minimize(f, (0, 0), method='SLSQP',constraints=restriccion).x

# graficamos
ig, ax = plt.subplots(figsize=(10, 8))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
ax.plot(x_, 1.75 + (x_-0.75)**4, 'k-', markersize=15)
ax.fill_between(x_, 1.75 + (x_-0.75)**4, 3, color='grey')
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
ax.set_ylim(-1, 3)
ax.set_xlabel(r"$x_0$", fontsize=18)
ax.set_ylabel(r"$x_1$", fontsize=18)
plt.colorbar(c, ax=ax)
plt.show()
