#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
import numpy as np

# the coefficient functions
def p(phi):
  sq = inner(phi.dx(0), phi.dx(0))
  aux = 4 / (4 - sq)
  truc = conditional(gt(sq, Constant(3)), Constant(4), aux)
  return truc

def q(phi):
  sq = inner(phi.dx(1), phi.dx(1))
  aux = 4 / sq
  truc = conditional(lt(sq, Constant(1)), Constant(4), aux)
  truc2 = conditional(gt(sq, Constant(4)), Constant(1), truc)
  return truc2

def v(phi):
  sq_x = inner(phi.dx(0), phi.dx(0))
  sq_y = inner(phi.dx(1), phi.dx(1))
  value = (1 - 0.25*sq_x) * sq_y
  return conditional(gt(abs(value), Constant(2*ln(2))), Constant(2*ln(2)), value)

def u(phi):
  prod = inner(phi.dx(0), phi.dx(1))
  return conditional(gt(abs(prod), Constant(2*sqrt(3))), Constant(2*sqrt(3)), prod)

# Create mesh and define function space
LL = 2 #length of rectangle
H = 1 #height of rectangle
#mesh= Mesh('mesh_1.msh')
size_ref = 25
mesh = RectangleMesh(size_ref, size_ref, LL, H, diagonal='crossed')
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)

#Dirichlet BC
# Boundary conditions
x = SpatialCoordinate(mesh)
#phi_D = as_vector((x[0], x[1], -x[0]))
phi_D = as_vector((x[0]*x[0], x[1]*x[1], -x[0]-x[1]))
bcs = [DirichletBC(V, phi_D, 1), DirichletBC(V, phi_D, 2), DirichletBC(V, phi_D, 3), DirichletBC(V, phi_D, 4)]

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
L = Constant(0) * psi[0] * dx
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

file_bis = File('laplacian.pvd')
proj = interpolate(phi - as_vector((x[0],x[1],0)), V)
file_bis.write(proj)
#sys.exit()

#Bilinear form
aux = as_vector((p(phi)*v(phi)*phi_t.dx(0) +  u(phi)*phi_t.dx(1), q(phi)*v(phi)*phi_t.dx(1) +  u(phi)*phi_t.dx(0)))
#a = inner(aux, grad(psi)) * dx
a = p(phi)*v(phi)*inner(phi_t.dx(0), psi.dx(0)) * dx + q(phi)*v(phi)*inner(phi_t.dx(1), psi.dx(1)) * dx + u(phi)*(inner(phi_t.dx(1), psi.dx(0))+inner(phi_t.dx(0), psi.dx(1))) * dx 

file = File('res.pvd')

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
for iter in range(maxiter):
  #linear solve
  A = assemble(a, bcs=bcs)
  b = assemble(L, bcs=bcs)
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
  
  #convergence test 
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  #output
  projected = Function(V, name='surface')
  projected.interpolate(phi - as_vector((x[0], x[1], 0)))
  file.write(projected)

  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))
sys.exit()

  
#Write 2d results
flat = File('flat_%i.pvd' % size_ref)
proj = project(phi, W, name='flat')
flat.write(proj)
  
#Write 3d results
file = File('new_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
projected = Function(W, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Test is inequalities are true
file_bis = File('verif_x.pvd')
proj = project(inner(phi.dx(0),phi.dx(0)), UU, name='test phi_x')
file_bis.write(proj)
file_ter = File('verif_y.pvd')
proj = project(inner(phi.dx(1),phi.dx(1)), UU, name='test phi_y')
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj = project(inner(phi.dx(0),phi.dx(1)), UU, name='test PS')
file_4.write(proj)
