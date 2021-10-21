#coding: utf-8

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm

#define values
theta = np.pi/2
L = 2*np.sin(0.5*np.arccos(0.5/np.cos(0.5*theta))) #length of rectangle
alpha = np.sqrt(1 / (1 - np.sin(theta/2)**2))
H = 2*np.pi/alpha #height of rectangle
l = np.sin(theta/2)*L #total height of cylindre
modif = 0.02 #0.02 #variation at the top


#writing the matrix of the system
A = np.zeros((6,6))
#Filling-in by line
A[0,:] = np.array([L**2/4, 0, 0, -L/2, 0, 1])
A[1,:] = np.array([L**2/4, H**2/4, L*H/4, L/2, H/2, 1])
A[2,:] = np.array([L**2/4, 0, 0, L/2, 0, 1])
A[3,:] = np.array([0, H**2/4, 0, 0, H/2, 1])
A[4,:] = np.array([L**2/4, H**2, -L*H/2, -L/2, H, 1])
A[5,:] = np.array([L**2/4, H**2, L*H/2, L/2, H, 1])

#Corresponding right-hand side
b = np.array([-l, l, l*(1+modif), 0, -l, l*(1-modif)])

#solution
coeffs = np.linalg.solve(A,b)
print(coeffs)

def f(x,y):
    return coeffs[0]*x*x + coeffs[1]*y*y + coeffs[2]*x*y + coeffs[3]*x + coeffs[4]*y + coeffs[5]
    

#test
x = np.linspace(-L/2, L/2, 30)
y = np.linspace(0, H, 30)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

#test
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


