#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys

# the coefficient functions
def p(phi):
  return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

# Create mesh and define function space
L = 2 #length of rectangle
H = 1 #height of rectangle #1.2 works #1.3 no
size_ref = 5 #degub: 5
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)

# Boundary conditions
x = SpatialCoordinate(mesh)
phi_D1 = as_vector((x[0], x[1], 0))
#modify this one to be the right BC
alpha = pi/4
l = H / sqrt(1 + H*H/L/L)
w = as_vector((L, 0, 0)) + H*H*L/(H*H+L*L)*as_vector((-L,H,0)) + l*as_vector((H/sqrt(H*H+L*L)*cos(alpha), L/sqrt(H*H+L*L)*cos(alpha), sin(alpha)))
u = -w + as_vector((L,0,0))
v = -w + as_vector((0,H,0))
phi_D2 = (1-x[0]/L)*u + (1-x[1]/H)*v + w

#test BC
f = Function(U)
f.interpolate(phi_D2)
f = project(f - as_vector((x[0], x[1], 0)), U)
file = File('test.pvd')
#file.write(f)
g = Function(U)
g.interpolate(Constant((0,0,0)))
file.write(g)
sys.exit()

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations

#Defining the bilinear forms
#bilinear form for linearization
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
L = pen/h**4 * inner(phi_D1, psi) *(ds(1)+ds(3)) + pen/h**4 * inner(phi_D2, psi) *(ds(2)+ds(4))

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
  pp = interpolate(p(phi), UU)
  PETSc.Sys.Print('Min of p: %.3e' % pp.vector().array().min())
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate

  ##ellipticity test
  #test = project(sq_norm(phi.dx(0)), UU)
  #with test.dat.vec_ro as v:
  #  value = v.max()[1]
  #try:
  #  assert value < 4 #not elliptic otherwise.
  #except AssertionError:
  #  PETSc.Sys.Print('Bouned slope condition %.2e' % value)
  
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
file = File('new_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
projected = project(phi - as_vector((x[0], x[1], 0)), U, name='surface')
file.write(projected)
