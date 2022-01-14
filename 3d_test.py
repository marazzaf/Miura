#coding: utf-8

import numpy as np
import sys
from mayavi import mlab

size_ref = 20
#data = np.loadtxt('hyperboloid_%i.txt' % size_ref)
#data = np.loadtxt('points_aux_%i.txt' % size_ref)
data = np.loadtxt('points_%i.txt' % size_ref)

x = data[:,0]
y = data[:,1]
z = data[:,2]

#test new plot
mlab.clf()
phi, theta = np.mgrid[0:np.pi:11j, 0:2*np.pi:11j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
mlab.mesh(x, y, z)
mlab.mesh(x, y, z, representation='wireframe', color=(0, 0, 0))
#mlab.mesh(x, y, z)
#mlab.mesh(x, y, z, representation='wireframe', color=(0, 0, 0))
