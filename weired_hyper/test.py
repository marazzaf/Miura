from firedrake import *
from firedrake.petsc import PETSc
import sys
import numpy as np

# Create mesh and define function space
size_ref = 3
L,H = 2,1
mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')
V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
f = Function(V, name='ref')
x = SpatialCoordinate(mesh)
f.interpolate(as_vector((exp(x[0]+x[1]), 1, 2)))
file = File('test_1.pvd')
file.write(f)

def func(data):
    res = np.zeros((len(data),3))
    for i,dat in enumerate(data):
        res[i,:] = f(dat)
    return res

#new mesh
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
W = VectorFunctionSpace(mesh, 'CG', 1)
X = interpolate(mesh.coordinates, W)
#print(X.dat.data_ro.shape)
#print('***************************')
#print(func(X.dat.data_ro).shape)
V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
F = Function(V, name='interpolated')
#print(F.dat.data[:].shape)
F.dat.data[:] = func(X.dat.data_ro)
file = File('test_2.pvd')
file.write(F)
