#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
import numpy as np
from comp_phi import comp_phi

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
mesh = Mesh('mesh_2.msh')
size_ref = 40 #degub: 5
#mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')
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
phi = as_vector((x[0], x[1], 0))
g_phi.interpolate(grad(phi_ref))
g_phi_t = TrialFunction(V)
g_psi = TestFunction(V)
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx #laplace in weak form
L = Constant(0) * g_psi[0,0] * dx

#Dirichlet BC other form
h = CellDiameter(mesh)
pen = 1e1
N = cross(g_phi[:,0], g_phi[:,1])
#bilinear
pen_term = pen/h**2 * inner(g_phi[:,0], g_phi_t[:,1]) * inner(g_phi[:,0], g_psi[:,1]) * ds + pen/h**2 * inner(g_phi[:,1], g_phi_t[:,0]) * inner(g_phi[:,1], g_psi[:,0]) * ds
pen_term += pen/h**2 * dot(g_phi_t[:,0], N) * dot(g_psi[:,0], N) * ds + pen/h**2 * dot(g_phi_t[:,1], N) * dot(g_psi[:,1], N) * ds 
pen_term += pen/h**2 * inner(g_phi[:,0], g_phi_t[:,0]) * inner(g_phi[:,0], g_psi[:,0]) * ds + pen/h**2 * inner(g_phi[:,1], g_phi_t[:,1]) * inner(g_phi[:,1], g_psi[:,1]) * ds
#linear
L = pen/h**2 * dot(grad(phi_ref)[:,0], N) * dot(g_psi[:,0], N) * ds + pen/h**2 * dot(grad(phi_ref)[:,1], N) * dot(g_psi[:,1], N) * ds
L += pen/h**2 * inner(grad(phi_ref)[:,0], grad(phi_ref)[:,0]) * inner(g_phi[:,0], g_psi[:,0]) * ds + pen/h**2 * inner(grad(phi_ref)[:,1], grad(phi_ref)[:,1]) * inner(g_phi[:,1], g_psi[:,1]) * ds

#Dirichlet BC
bcs = [DirichletBC(V, grad(phi_ref), 6), DirichletBC(V, grad(phi_ref), 8)]
#bcs = [DirichletBC(V, grad(phi_ref), 1), DirichletBC(V, grad(phi_ref), 2), DirichletBC(V, grad(phi_ref), 3), DirichletBC(V, grad(phi_ref), 4)]
#bcs = [DirichletBC(V, grad(phi_ref), 1), DirichletBC(V, grad(phi_ref), 2)]

#solving
A = assemble(laplace+pen_term, bcs=bcs)
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

##Neumann-type BC
#h = CellDiameter(mesh)
#pen = 1e1
#N = cross(g_phi[:,0], g_phi[:,1])
#pen_term = pen/h**2 * inner(g_phi[:,0], g_phi[:,1]) * inner(g_phi[:,0], g_psi[:,1]) * ds + pen/h**2 * inner(g_phi[:,0], g_phi[:,1]) * inner(g_phi[:,1], g_psi[:,0]) * ds
#pen_term += pen/h**2 * dot(g_phi[:,0], N) * dot(g_psi[:,0], N) * ds + pen/h**2 * dot(g_phi[:,1], N) * dot(g_psi[:,1], N) * ds 
#pen_term += pen/h**2 * inner(g_phi[:,0], g_phi[:,0]) * inner(g_phi[:,0], g_psi[:,0]) * ds + pen/h**2 * inner(g_phi[:,1], g_phi[:,1]) * inner(g_phi[:,1], g_psi[:,1]) * ds
#L = pen/h**2 * dot(grad(phi_ref)[:,0], N) * dot(g_psi[:,0], N) * ds + pen/h**2 * dot(grad(phi_ref)[:,1], N) * dot(g_psi[:,1], N) * ds
#L += pen/h**2 * inner(grad(phi_ref)[:,0], grad(phi_ref)[:,0]) * inner(g_phi[:,0], g_psi[:,0]) * ds + pen/h**2 * inner(grad(phi_ref)[:,1], grad(phi_ref)[:,1]) * inner(g_phi[:,1], g_psi[:,1]) * ds

##Dirichlet BC
#pen_term = pen/h**2 * inner(g_phi, g_psi) * ds
#L = pen/h**2 * inner(grad(phi_ref), g_psi) * ds

#Dirichlet BC
bcs = [DirichletBC(V, grad(phi_ref), 6), DirichletBC(V, grad(phi_ref), 8)]

# Solving with Newton method
#solve(a == 0, g_phi, bcs=bcs, solver_parameters={'snes_monitor': None})
#solve(a - L == 0, g_phi, solver_parameters={'snes_monitor': None})

##Computing the error
#err = sqrt(assemble(inner(g_phi - grad(phi_ref), g_phi - grad(phi_ref)) * dx))
#PETSc.Sys.Print('Error: %.3e' % err)
#
#file = File('res.pvd')
#file.write(g_phi)
#sys.exit()

#bilinear form for linearization
a = inner(p(g_phi) * g_phi_t[:,0].dx(0) + q(g_phi) * g_phi_t[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx

#Dirichlet BC
pen_term = pen/h**2 * inner(g_phi_t, g_psi) * (ds(1) + ds(2))
L = pen/h**2 * inner(grad(phi_ref), g_psi) * (ds(1) + ds(2))

#Dirichlet BC other form
#bilinear
pen_term = pen/h**2 * inner(g_phi[:,0], g_phi_t[:,1]) * inner(g_phi[:,0], g_psi[:,1]) * ds + pen/h**2 * inner(g_phi[:,1], g_phi_t[:,0]) * inner(g_phi[:,1], g_psi[:,0]) * ds
pen_term += pen/h**2 * dot(g_phi_t[:,0], N) * dot(g_psi[:,0], N) * ds + pen/h**2 * dot(g_phi_t[:,1], N) * dot(g_psi[:,1], N) * ds 
pen_term += pen/h**2 * inner(g_phi[:,0], g_phi_t[:,0]) * inner(g_phi[:,0], g_psi[:,0]) * ds + pen/h**2 * inner(g_phi[:,1], g_phi_t[:,1]) * inner(g_phi[:,1], g_psi[:,1]) * ds
#linear
L = pen/h**2 * dot(grad(phi_ref)[:,0], N) * dot(g_psi[:,0], N) * ds + pen/h**2 * dot(grad(phi_ref)[:,1], N) * dot(g_psi[:,1], N) * ds
L += pen/h**2 * inner(grad(phi_ref)[:,0], grad(phi_ref)[:,0]) * inner(g_phi[:,0], g_psi[:,0]) * ds + pen/h**2 * inner(grad(phi_ref)[:,1], grad(phi_ref)[:,1]) * inner(g_phi[:,1], g_psi[:,1]) * ds

#For output of phi
W = VectorFunctionSpace(mesh, "CG", 3, dim=3)

# Picard iteration
tol = 1e-5
maxiter = 50
phi_old = Function(V) #for iterations
file = File('res.pvd')
projected = Function(W, name='surface')
for iter in range(maxiter):
  #linear solve
  A = assemble(a + pen_term, bcs=bcs)
  b = assemble(L, bcs=bcs)
  solve(A, g_phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate

  #Compute phi
  phi = comp_phi(mesh, g_phi)
  projected.interpolate(phi - as_vector((x[0], x[1], 0)))
  file.write(projected)
    
  eps = sqrt(assemble(inner(grad(g_phi-phi_old), grad(g_phi-phi_old))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H1 seminorm of delta: {:10.2e}'.format(iter+1, eps))

  if eps < tol:
    break
  phi_old.assign(g_phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

  
err = sqrt(assemble(inner(g_phi - grad(phi_ref), g_phi - grad(phi_ref)) * dx))
PETSc.Sys.Print('Error: %.3e' % err)
