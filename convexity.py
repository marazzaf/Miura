#coding: utf-8

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

x = np.arange(0, np.sqrt(3), 1e-3)
y = np.arange(1, 2, 1e-3)

x,y = np.meshgrid(x, y)
#z = ((1 - 0.25*x*x)*y*y - 1)**2
z = (1 - 0.25*x*x)*y*y - 1

print(np.amin(z))
nz = np.where(abs(z) < 1.e-7)
print(x[nz])
print(y[nz])
#sys.exit()

plt.plot(x, np.sqrt(1 / (1 - 0.25*x*x)), 'x')
plt.show()
sys.exit()

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
