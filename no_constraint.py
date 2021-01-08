#coding: utf-8

from dolfin import *
import numpy as np
import ufl

size_ref = 5 #debug
L,l = 1,1
mesh = RectangleMesh(Point(-L/2,-l/2), Point(L/2, l/2), size_ref, size_ref, "crossed")
bnd = MeshFunction('size_t', mesh, 1)
bnd.set_all(0)

V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)

phi_x = Function(V)
phi_y = Function(V)
phi = as_vector((phi_x, phi_y))
#phi = Function(W)
v = TestFunction(V)

#Dirichlet BC
R = 10 #radius of the outer circle
x = SpatialCoordinate(mesh)
theta = ufl.atan_2(x[0], x[1])
#phi = Expression(('R*cos(', 'R*'), R=R, degree = 2)
phi_D = as_vector((R*cos(theta), R*sin(theta)))

#creating the bc object
#bc = DirichletBC(W, phi_D, bnd, 0)
bc_1 = DirichletBC(V, phi_D[0], bnd, 0)
bc_2 = DirichletBC(V, phi_D[1], bnd, 0)
bc = [bc_1, bc_2]

#Writing energy. No constraint for now...
norm_phi_x = sqrt(inner(phi.dx(0), phi.dx(0)))
norm_phi_y = sqrt(inner(phi.dx(1), phi.dx(1)))
#f = as_vector((2*np.arctanh(0.5*norm_phi_x), -4/norm_phi_y)) #how can I create a ufl function computing arctanh?
f = as_vector((2*ufl.atan(0.5*norm_phi_x), -4/norm_phi_y))


#bilinear form
a = inner(f, grad(v)) * dx

#solving problem
solve(a == 0, phi_x, bc, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

