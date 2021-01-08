#coding: utf-8

from dolfin import *
import numpy as np
import ufl
import sys

size_ref = 5 #debug
L,l = 1,1
mesh = RectangleMesh(Point(-L/2,-l/2), Point(L/2, l/2), size_ref, size_ref, "crossed")
bnd = MeshFunction('size_t', mesh, 1)
bnd.set_all(0)

#V = FunctionSpace(mesh, 'CG', 1)
#W = VectorFunctionSpace(mesh, 'CG', 1)

#Créer un espace mixte avec V et W pour pouvoir résoudre?
Ve = VectorElement('CG',  mesh.ufl_cell(), 1)
We = FiniteElement('CG',  mesh.ufl_cell(), 1)
M = MixedElement(Ve, We)
M = FunctionSpace(mesh, M)
V = M.sub(0).collapse()
W = M.sub(1).collapse()

#Creating the functions to define the bilinear form
#tot = Function(M)
#phi,truc = tot.split()
#bidule,psi = TestFunctions(M)
phi = Function(V)
psi = TestFunction(W)

#Dirichlet BC
R = 10 #radius of the outer circle
x = SpatialCoordinate(mesh)
theta = ufl.atan_2(x[0], x[1])
#phi_D = Expression(('R*cos(', 'R*'), R=R, degree = 2)
phi_D = as_vector((R*cos(theta), R*sin(theta)))
#test = Expression('atanh(x[0])', degree=3) #atanh bien def en C
#comment s'en servir pour avoir la fonction ufl?

#creating the bc object
bc = DirichletBC(V, phi_D, bnd, 0)
#bc_1 = DirichletBC(M.sub(0).sub(0), phi_D[0], bnd, 0)
#bc_2 = DirichletBC(M.sub(0).sub(1), phi_D[1], bnd, 0)
#bc = [bc_1, bc_2]

#Writing energy. No constraint for now...
norm_phi_x = sqrt(inner(phi.dx(0), phi.dx(0)))
norm_phi_y = sqrt(inner(phi.dx(1), phi.dx(1)))
#f = as_vector((2*np.arctanh(0.5*norm_phi_x), -4/norm_phi_y)) #how can I create a ufl function computing arctanh?
f = as_vector((2*ufl.atan(0.5*norm_phi_x), -4/norm_phi_y))


#bilinear form
#a = inner(f, grad(psi)) * dx
#a = inner(phi, grad(psi)) * dx - psi*dx
a = inner(as_vector((2*ufl.atan(0.5*norm_phi_x),norm_phi_y)), grad(psi)) * dx - psi * dx

#solving problem
solve(a == 0, phi, bc, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

