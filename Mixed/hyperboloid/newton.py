#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
import numpy as np
sys.path.append('..')
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

#Creating mesh
size_ref = 25 #25, 50, 100, 200
mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')
h = max(L/size_ref, H/size_ref)
PETSc.Sys.Print('Mesh size: %.5e' % h)

#Function Space
W = TensorFunctionSpace(mesh, "CG", 1, shape=(3,2))
Q = VectorFunctionSpace(mesh, "CG", 1, dim=3)
V = W * Q
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#  Ref solution
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_ref = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

#initial guess
#solve laplace equation on the domain
v = Function(V, name='grad solution')
g_phi_t,q_t = TrialFunctions(V)
g_psi,r = TestFunctions(V)
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx + 10 * inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx #laplace in weak form
#Add somthing for the constraint...
laplace += inner(g_psi[:,0].dx(1) - g_psi[:,1].dx(0), q_t) * dx + inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), r) * dx
L = Constant(0) * g_psi[0,0] * dx

#Dirichlet BC
bcs = [DirichletBC(V.sub(0), grad(phi_ref), 1), DirichletBC(V.sub(0), grad(phi_ref), 2)]
#How to impose zero average for the Lagrange multiplier

#solving
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(A, v, b, solver_parameters={'direct_solver': 'mumps'})
g_phi,qq = v.split()
#solve(A, phi, b, solver_parameters={'ksp_type': 'cg','pc_type': 'bjacobi', 'ksp_rtol': 1e-5})
PETSc.Sys.Print('Laplace equation ok')

#Write 3d results
file = File('laplacian.pvd')
phi = comp_phi(mesh, g_phi)
W = VectorFunctionSpace(mesh, "CG", 1, dim=3)
projected = Function(W, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#bilinear form for linearization
v = Function(V, name='grad solution')
g_phi,qq = split(v)
#qq = as_vector((v[6], v[7], v[8]))
#gg_phi = as_tensor(((v[0], v[1]), (v[2], v[3]), (v[4], v[5])))

#a = inner(p(gg_phi) * gg_phi[:,0].dx(0) + q(gg_phi) * gg_phi[:,1].dx(1), g_psi[:,0]) * dx
a = inner(p(g_phi) * g_phi[:,0].dx(0) + q(g_phi) * g_phi[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += 10 * inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx #rot stabilization
a += inner(g_psi[:,0].dx(1) - g_psi[:,1].dx(0), qq) * dx + inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), r) * dx #for mixed form

# Solving with Newton method
solve(a == 0, v, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})
g_phi,qq = v.split()

#Compute phi
phi = comp_phi(mesh, g_phi)
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file = File('res_newton.pvd')
file.write(projected)
  
err = errornorm(grad(phi_ref), g_phi, 'l2')
PETSc.Sys.Print('H1 error: %.3e' % err)

sys.exit()

vol = assemble(Constant(1) * dx(mesh))
mean = Constant((assemble(phi[0] / vol * dx), assemble(phi[1] / vol * dx), assemble(phi[2] / vol * dx)))

phi_mean = Function(W)
phi_mean.interpolate(phi - mean)
err = errornorm(phi_ref, phi_mean, 'l2')
PETSc.Sys.Print('L2 error: %.3e' % err)

WW = FunctionSpace(mesh, 'CG', 1)
u = Function(WW, name='u')
u.interpolate(inner(g_phi[:,0], g_phi[:,1]))
file = File('u.pvd')
file.write(u)
err = errornorm(Constant(0), u, 'l2')
PETSc.Sys.Print('L2 error: %.3e' % err)

v = Function(WW, name='v')
aux = 1 - 0.25 * inner(g_phi[:,0], g_phi[:,0])
v.interpolate(ln(aux * inner(g_phi[:,1], g_phi[:,1])))
file = File('v.pvd')
file.write(v)
err = errornorm(Constant(0), v, 'l2')
PETSc.Sys.Print('L2 error: %.3e' % err)

x = Function(WW, name='phi_x')
x.interpolate(inner(g_phi[:,0], g_phi[:,0]))
file = File('phi_x.pvd')
file.write(x)
y = Function(WW, name='phi_y')
y.interpolate(inner(g_phi[:,1], g_phi[:,1]))
file = File('phi_y.pvd')
file.write(y)
