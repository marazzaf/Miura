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

# Create mesh
L = 2 #length of rectangle
H = 1 #height of rectangle #1.2 works #1.3 no
size_ref = 1 #50
#mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
mesh = Mesh('mesh_%i.msh' % size_ref)
#Try to get a unit disk mesh?

# Define function space
V = TensorFunctionSpace(mesh, "CG", 1, shape=(3,2))
PETSc.Sys.Print('Nb dof: %i' % V.dim())
W = VectorFunctionSpace(mesh, "CG", 1, dim=3)
projected = Function(W, name='surface')
WW = FunctionSpace(mesh, 'CG', 1)

#Ref solution
x = SpatialCoordinate(mesh)

#previous ref
alpha = pi/2 #pi/4
l = H*L / sqrt(L*L + H*H)
sin_gamma = H / sqrt(L*L+H*H)
cos_gamma = L / sqrt(L*L+H*H)
DB = l*as_vector((sin_gamma,cos_gamma,0))
DBp = l*sin(alpha)*Constant((0,0,1)) + cos(alpha) * DB
OC = as_vector((L, 0, 0))
CD = Constant((-sin_gamma*l,H-cos_gamma*l,0))
OBp = OC + CD + DBp

phi_ref = (OBp - OC - as_vector((0, H, 0))) *  x[0]*x[1]/L/H + as_vector((x[0], 0, 0)) + as_vector((0, x[1], 0))

# Boundary conditions
phi_x = phi_ref.dx(0)
phi_y = phi_ref.dx(1)
N = cross(phi_x, phi_y) / norm(cross(phi_x, phi_y))
r = sqrt(x[0]**2 + x[1]**2)
theta = atan(x[1]/(x[0]+.0001))
e_r = as_vector((cos(theta), sin(theta), 0))
#e_theta = as_vector((-sin(theta), cos(theta), 0))
n_G_x = r/2
n_G_y = 4/(4-n_G_x**2)
G_x = n_G_x * e_r
G_y = n_G_y * cross(N, e_r)
G = as_tensor((G_x, G_y)).T

#test
aux = Function(W)
file = File('test_phi_x.pvd')
aux.interpolate(G_x)
file.write(aux)
aux = Function(WW)
file = File('test_phi_y.pvd')
aux.interpolate(norm(cross(N, e_r)))
file.write(aux)
sys.exit()

#initial guess
#solve laplace equation on the domain
g_phi = Function(V, name='grad solution')
g_phi_t = TrialFunction(V)
g_psi = TestFunction(V)
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx #laplace in weak form
L = Constant(0) * g_psi[0,0] * dx

#Dirichlet BC
bcs = [DirichletBC(V, G, 1), DirichletBC(V, G, 2), DirichletBC(V, G, 3), DirichletBC(V, G, 4)]

#solving
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(A, g_phi, b, solver_parameters={'direct_solver': 'mumps'})
#solve(A, phi, b, solver_parameters={'ksp_type': 'cg','pc_type': 'bjacobi', 'ksp_rtol': 1e-5})
PETSc.Sys.Print('Laplace equation ok')

##Write 3d results
#file = File('laplacian.pvd')
#phi = comp_phi(mesh, g_phi)
#projected.interpolate(phi - as_vector((x[0], x[1], 0)))
#file.write(projected)

#bilinear form for linearization
a = inner(p(g_phi) * g_phi[:,0].dx(0) + q(g_phi) * g_phi[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx

# Solving with Newton method
solve(a == 0, g_phi, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

#Compute phi
phi = comp_phi(mesh, g_phi)
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file = File('res_newton.pvd')
file.write(projected)
  
#err = errornorm(grad(phi_ref), g_phi, 'l2')
#PETSc.Sys.Print('H1 error: %.3e' % err)
#
#vol = assemble(Constant(1) * dx(mesh))
#mean = Constant((assemble(phi[0] / vol * dx), assemble(phi[1] / vol * dx), assemble(phi[2] / vol * dx)))
#
#phi_mean = Function(W)
#phi_mean.interpolate(phi - mean)
#err = errornorm(phi_ref, phi_mean, 'l2')
#PETSc.Sys.Print('L2 error: %.3e' % err)

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
