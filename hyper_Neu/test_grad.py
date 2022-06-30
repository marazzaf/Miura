#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
import numpy as np

# the coefficient functions
def p(g_phi):
  sq = inner(g_phi[:,0], g_phi[:,0])
  aux = 4 / (4 - sq)
  truc = conditional(gt(sq, Constant(3)), Constant(4), aux)
  return truc

def q(g_phi):
  sq = inner(g_phi[:,1], g_phi[:,1])
  aux = 4 / sq
  truc = conditional(lt(sq, Constant(1)), Constant(4), aux)
  truc2 = conditional(gt(sq, Constant(4)), Constant(1), truc)
  return truc2

# Size for the domain
theta = pi/2 #pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle
l = sin(theta/2)*L

#Creating mesh
#mesh = Mesh('mesh_1.msh')
size_ref = 20 #degub: 5
mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')
V = TensorFunctionSpace(mesh, "CG", 2, shape=(3,2))
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#  Ref solution
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_ref = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

#initial guess
#solve laplace equation on the domain
g_phi = Function(V, name='grad solution')
g_phi_t = TrialFunction(V)
g_psi = TestFunction(V)
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx #laplace in weak form
L = Constant(0) * g_psi[0,0] * dx

#Dirichlet BC
bcs = [DirichletBC(V, grad(phi_ref), 1), DirichletBC(V, grad(phi_ref), 2)]

#solving
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(A, g_phi, b, solver_parameters={'direct_solver': 'mumps'})
#solve(A, phi, b, solver_parameters={'ksp_type': 'cg','pc_type': 'bjacobi', 'ksp_rtol': 1e-5})
PETSc.Sys.Print('Laplace equation ok')

#Write 3d results
file = File('laplacian.pvd')
file.write(g_phi)

#ref
file = File('ref.pvd')
ref = Function(V, name='grad ref')
ref.interpolate(grad(phi_ref))
file.write(ref)

#Writing our problem now
#bilinear form for linearization
a = inner(p(g_phi) * g_phi[:,0].dx(0) + q(g_phi) * g_phi[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx

# Solving with Newton method
solve(a == 0, g_phi, bcs=bcs, solver_parameters={'snes_monitor': None})

#Computing the error
err = sqrt(assemble(inner(g_phi - grad(phi_ref), g_phi - grad(phi_ref)) * dx))
PETSc.Sys.Print('Error: %.3e' % err)

file = File('res.pvd')
file.write(g_phi)
sys.exit()

#file = File('res.pvd')
#projected = Function(U, name='surface')
#projected.interpolate(phi - as_vector((x[0], x[1], 0)))
#file.write(projected)
#sys.exit()

#bilinear form for linearization
a = inner(p(g_phi) * g_phi_t[:,0].dx(0) + q(g_phi) * g_phi_t[:,1].dx(1),  g_psi[:,0]) * dx
a += inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), g_psi[:,1]) * dx
L = Constant(0) * g_psi[0,0] * dx

# Picard iteration
tol = 1e-5
maxiter = 50
phi_old = Function(V) #for iterations
for iter in range(maxiter):
  #linear solve
  A = assemble(a, bcs=bcs)
  b = assemble(L, bcs=bcs)
  solve(A, g_phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
    
  eps = sqrt(assemble(inner(grad(g_phi-phi_old), grad(g_phi-phi_old))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))

  if eps < tol:
    break
  phi_old.assign(g_phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Computing error
X = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
projected = interpolate(div(grad(phi)), X)
ref = interpolate(div(grad(phi_ref)), X)
err = sqrt(assemble(inner(div(grad(phi-phi_ref)), div(grad(phi-phi_ref)))*dx))
PETSc.Sys.Print('Error: %.3e' % err)

