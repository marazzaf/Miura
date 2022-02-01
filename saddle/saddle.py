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
size_ref = 20 #degub: 5
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)

# Boundary conditions
beta = 0.1
x = SpatialCoordinate(mesh)
phi_D1 = beta*as_vector((x[0], x[1], 0))

#modify this one to be the right BC
alpha = 0
#L,H = beta*L,beta*H
l = H*L / sqrt(L*L + H*H)
sin_gamma = H / sqrt(L*L+H*H)
cos_gamma = L / sqrt(L*L+H*H)
DB = l*as_vector((sin_gamma,cos_gamma,0))
DBp = l*sin(alpha)*Constant((0,0,1)) + cos(alpha) * DB
OC = as_vector((L, 0, 0))
CD = Constant((-sin_gamma*l,H-cos_gamma*l,0))
OBp = OC + CD + DBp
BpC = -DBp - CD
BpA = BpC + Constant((-L, H, 0))
phi_D2 = (1-x[0]/L)*BpC + (1-x[1]/H)*BpA + OBp

##test BC
#f = Function(U)
#f.interpolate(phi_D2)
#file = File('test.pvd')
#file.write(f)
#file_3 = File('surf.pvd')
#file_3.write(project(f- as_vector((x[0], x[1], 0)), U))
#file_2 = File('test_2.pvd')
#g = Function(U)
#g.interpolate(Constant((0,0,0)))
#file_2.write(g)
#sys.exit()

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
#laplace = inner(div(grad(phi_t)), div(grad(psi))) * dx #test
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#test of coeff
file_test = File('test.pvd')
pp = Function(UU, name='coef')

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
for iter in range(maxiter):
  #linear solve
  A = assemble(a)
  b = assemble(L)
  pp.interpolate(p(phi))
  file_test.write(pp, time=iter)
  PETSc.Sys.Print('Min of p: %.3e' % pp.vector().array().min())
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate

  ##ellipticity test
  #test = project(sq_norm(phi.dx(0)), UU)
  #with test.dat.vec_ro as v:
  #  value = v.max()[1]
  #assert value < 4, ('Bouned slope condition %.2e' % value)
  
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
