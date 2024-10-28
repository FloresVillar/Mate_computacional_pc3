import numpy as np

# Función objetivo
def f(x):
    return x**2

# Restricción g(x) = 1 - x <= 0
def g(x):
    return 1 - x

# Función de barrera
def B(x):
    if g(x)>0:
        return -1 / g(x)
    else:
        return np.inf
# Función penalizada
def phi(x, mu):
    return f(x) + 1/mu * B(x)

# Gradiente de la función penalizada
def grad_phi(x, mu):
    if x>=1:
        return 2 * x + 1 / (mu * (1 - x)**2)
    else:
        return np.inf
# Descenso por gradiente para minimizar la función penalizada
def grad_desc_for_penalizacion(x, mu, lr=0.1, nMax=100, epsilon=0.0001):
    i = 0
    gr = grad_phi(x, mu)  
    while i < nMax and np.linalg.norm(gr) >= epsilon:
        x = x - lr * gr  # Actualización del gradiente
        gr = grad_phi(x, mu)  # Recalcular el gradiente
        i += 1
    return x

# Método de puntos interiores
def puntos_interiores(x, mu=0.1, epsilon=0.01, beta=10, nMax=100):
    delta = np.inf
    i = 0
    while i < nMax and delta > epsilon:
        x1 = grad_desc_for_penalizacion(x, mu)  # Minimización
        delta = np.linalg.norm(x1 - x)  # Cambio entre iteraciones
        x = x1
        mu = mu * beta  # Incremento de mu
        i += 1
    return x

if __name__ == '__main__':
    x0 = 0.5  # Valor inicial
    mu = 1  # Valor inicial de mu (pequeño para empezar)
    beta = 10  # Factor de crecimiento de mu (mayor que 1)
    epsilon = 0.0001  # Umbral de convergencia
    nMax = 50  # Máximo número de iteraciones

    # Ejecutar el método de puntos interiores
    x_min = puntos_interiores(x0, mu, epsilon, beta, nMax)

    # Imprimir el resultado
    print("El valor mínimo encontrado es:")
    print(x_min)
