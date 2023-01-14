import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#def exponential_decay(t, y): return -0.5 * y
#
#sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
#
#print(sol.t)
#print(sol.y)

alpha = 1

def rhs(t, y):
    aux = 4*alpha*alpha * y[0] / (4 - alpha*alpha*y[0]*y[0])**2
    return [y[1], aux]

rho = solve_ivp(rhs, [0, 10], [0, 0.5])

plt.plot(rho.t, rho.y[0], '*-')
plt.show()

L = 2*np.pi/alpha
