import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys

alpha = 1 #np.sqrt(3) / 2

def rhs(t, y):
    aux = 4*alpha*alpha * y[0] / (4 - alpha*alpha*y[0]*y[0])**2
    return [y[1], aux]

N = 50
L = 2*np.pi/alpha
H = 4 #1.3

beta_0 = 0
theta_0 = np.pi/4
rho_0 = 0.1 #0
rho_p_0 = 2*np.sin(beta_0)*np.cos(theta_0/2)
rho = solve_ivp(rhs, [0, H], [rho_0, rho_p_0], max_step=H/N)

plt.plot(rho.t, rho.y[0], '*-')
plt.show()
sys.exit()

M = len(rho.y[0])
print(M)

aux = np.cos(beta_0)/np.cos(theta_0/2)
Z = np.arange(0, aux*H, aux*H/M)
print(len(Z))
X = np.arange(0, L, L/M)
print(len(X))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for r,z in zip(rho.y[0],Z):
    for x in X:
        ax.scatter(r*np.cos(alpha*x), r*np.sin(alpha*x), z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
