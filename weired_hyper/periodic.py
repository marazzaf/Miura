#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys

# the coefficient functions
def p(phi):
  aux = 1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))
  return interpolate(conditional(lt(aux, Constant(1)), Constant(100), aux), UU)

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

#geometric parameters
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle
l = sin(theta/2)*L #total height of cylindre
modif = 0.5 #0.1 #variation at the top

# Create mesh and define function space
size_ref = 20 #degub: 5
mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)

# Boundary conditions
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(L/2)**2 + 1)
#BC on lower part
phi_D1 = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), -2*sin(theta/2)*L))
#BC on upper part
modif = 2
#z = -l*modif*(x[1]-H/2)**2*4/H/H + l*(1+modif)
z = l*modif*(x[1]-H/2)**2*4/H/H + l*(1+modif)
phi_D2 = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))


# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations

#Defining the bilinear forms for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1
#lhs
pen_term = pen/h**4 * inner(phi_t, psi) * ds
a += pen_term
#rhs
L = pen/h**4 * inner(phi_D1, psi) * ds(1) + pen/h**4 * inner(phi_D2, psi) * ds(2)

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
for iter in range(maxiter):
  #linear solve
  A = assemble(a)
  b = assemble(L)
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
  
  #convergence test 
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Write 2d results
flat = File('flat_%i.pvd' % size_ref)
W = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
proj = project(phi, W, name='flat')
flat.write(proj)
  
#Write 3d results
file = File('new_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
projected = Function(W, name='surface')
cond = conditional(gt(x[1], H-H/100), 0, x[1]) #conditional(lt(x[1], H/100), 0, x[1]) or
aux = as_vector((x[0], x[1], 0)) #cond, 0))
#cond_phi_y = conditional(gt(phi[1], H-H/100), 0, x[1])
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
##special = interpolate(phi[1] - x[1], UU)
##print(special((2*sin(0.5*acos(0.5/cos(0.5*theta)))/2,H)))
##print(special((2*sin(0.5*acos(0.5/cos(0.5*theta)))/2,0)))
##sys.exit()
##import numpy as np
##a = special.vector().array()
##w = np.where(a > 0.2)
##print(a[w])
##special.vector().array()[w] = 0.2*np.ones_like(w)
##file.write(special)
##sys.exit()
#other = Function(W, name='surface')
#other.interpolate(as_vector((projected[0], conditional(gt(projected[1], 0), 0, projected[1]), projected[2])))
#file.write(other)
#sys.exit()
file.write(projected)
