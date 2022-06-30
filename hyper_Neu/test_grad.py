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
g_phi = Function(V, name='solution')
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

#Reconstructing phi
W = VectorFunctionSpace(mesh, "CG", 3, dim=3)
phi = Function(W, name='solution')
phi_t = TrialFunction(W)
aux = inner(grad(phi_t), g_psi) * dx
L = inner(g_phi, g_psi) * dx

#BC for now. Make zero average later
bcs_test = [DirichletBC(W, phi_ref, 1), DirichletBC(W, phi_ref, 2)]

#solving
A = assemble(A) #, bcs=bcs_test)
b = assemble(L) #, bcs=bcs_test)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})

#Write 3d results
file = File('laplacian.pvd')
projected = Function(W, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)
sys.exit()

#Writing our problem now
#bilinear form for linearization
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi)) 
#a = Gamma * inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#New bilinear forms
a = Gamma * (p(phi) * dot(phi.dx(0).dx(0), N) + q(phi)*dot(phi.dx(1).dx(1), N)) * div(grad(psi[2])) * dx
#a += Gamma * (p(phi) * dot(phi_t.dx(0).dx(0), phi.dx(0)) + q(phi)*dot(phi_t.dx(1).dx(1), phi.dx(0))) * div(grad(psi[0])) * dx
a += (p(phi) * u.dx(0) + 2*v.dx(1)) * psi[0] * dx
#a += Gamma * (p(phi) * dot(phi_t.dx(0).dx(0), phi.dx(1)) + q(phi)*dot(phi_t.dx(1).dx(1), phi.dx(1))) * div(grad(psi[1])) * dx
a += (p(phi) * u.dx(1) - 2*v.dx(0)) * psi[1] * dx

#New pen term?
pen_term = pen/h**4 * inner(phi, psi) * (ds(1) + ds(2))
L = pen/h**4 * inner(phi_ref, psi) * (ds(1) + ds(2))

# Solving with Newton method
solve(a+pen_term - L == 0, phi, solver_parameters={'snes_monitor': None})
file = File('res.pvd')
projected = Function(U, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)
sys.exit()

# Picard iteration
tol = 1e-5
maxiter = 50
phi_old = Function(V) #for iterations
for iter in range(maxiter):
  #linear solve
  A = assemble(a+pen_term)
  b = assemble(L)
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
    
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  projected = Function(U, name='surface')
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
#For projection
file = File('res.pvd')
projected = Function(U, name='surface')
#projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)
sys.exit()

#Computing error
X = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
projected = interpolate(div(grad(phi)), X)
ref = interpolate(div(grad(phi_ref)), X)
err = sqrt(assemble(inner(div(grad(phi-phi_ref)), div(grad(phi-phi_ref)))*dx))
#PETSc.Sys.Print('Error: %.3e' % err)

#Tests if inequalities are true
file_bis = File('verif_x.pvd')
phi_x = interpolate(phi.dx(0), U)
proj = project(inner(phi_x,phi_x), UU, name='test phi_x')
file_bis.write(proj)
file_ter = File('verif_y.pvd')
phi_y = interpolate(phi.dx(1), U)
proj = project(inner(phi_y,phi_y), UU, name='test phi_y')
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj = project(inner(phi_x,phi_y), UU, name='test PS')
file_4.write(proj)

