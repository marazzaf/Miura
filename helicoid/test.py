#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys

# the coefficient functions
def p(phi):
  sq = inner(phi.dx(0), phi.dx(0))
  aux = 4 / (4 - sq )
  return conditional(gt(sq, Constant(4)), Constant(100), aux)

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

# Create mesh and define function space
L = 2*pi #length of rectangle
H = 2 #height of rectangle
size_ref = 50
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)
W = VectorFunctionSpace(mesh, 'CG', 4, dim=3)

# Boundary conditions
x = SpatialCoordinate(mesh)
phi_D = x[1]*as_vector((cos(x[0]), sin(x[0]), 0)) + as_vector((0,0,x[0]))

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations

#Defining the bilinear forms
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi))
a = Gamma * inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1
#lhs
pen_term = pen/h**4 * inner(phi_t, psi) * ds
a += pen_term
#rhs
L = pen/h**4 * inner(phi_D, psi) * ds

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
#laplace = inner(div(grad(phi_t)), div(grad(psi))) * dx #test
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#write laplace
file = File('laplace_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
projected = Function(W, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#test
eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
PETSc.Sys.Print('Before computation  H2 seminorm of delta: {:10.2e}'.format(eps))

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 100
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
phi_x = interpolate(phi.dx(0), W)
proj = project(inner(phi_x,phi_x), UU, name='test phi_x')
file_bis.write(proj)
file_ter = File('verif_y.pvd')
phi_y = interpolate(phi.dx(1), W)
proj = project(inner(phi_y,phi_y), UU, name='test phi_y')
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj = project(inner(phi_x,phi_y), UU, name='test PS')
file_4.write(proj)

#Test
test = project(div(grad(phi)), W, name='minimal')
file_6 = File('minimal_bis.pvd')
file_6.write(test)
