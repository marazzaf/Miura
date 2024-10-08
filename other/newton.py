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
size_ref = 50 #25, 50, 100, 200
mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')
h = max(L/size_ref, H/size_ref)
PETSc.Sys.Print('Mesh size: %.5e' % h)

#Function Space
W = TensorFunctionSpace(mesh, "CG", 2, shape=(3,2))
Q = VectorFunctionSpace(mesh, "CG", 1, dim=3)
V = W * Q
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#  Ref solution
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_ref = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))
G1 = grad(phi_ref)
#How to rotate it on one side?
alpha = np.pi/6 #1e-1
rot = as_tensor(((1, 0, 0), (0, cos(alpha), -sin(alpha)), (0, sin(alpha), cos(alpha))))
G2 = dot(rot, grad(phi_ref))

#initial guess
#solve laplace equation on the domain
g_phi_t,q_t = TrialFunctions(V)
g_psi,r = TestFunctions(V)
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx + 10 * inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx #laplace in weak form
laplace += inner(g_psi[:,0].dx(1) - g_psi[:,1].dx(0), q_t) * dx + inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), r) * dx
L = Constant(0) * g_psi[0,0] * dx

#Dirichlet BC
bcs = [DirichletBC(V.sub(0), G1, 1), DirichletBC(V.sub(0), G2, 2)]
#How to impose zero average for the Lagrange multiplier

#solving
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
v = Function(V, name='grad solution')
solve(A, v, b, solver_parameters={'direct_solver': 'mumps'})
#solve(A, v, b, solver_parameters={'ksp_type': 'cg','pc_type': 'gamg', 'ksp_rtol': 1e-5})
#g_phi = v.sub(0)
#qq = v.sub(1)
g_phi,qq = split(v)
PETSc.Sys.Print('Laplace equation ok')

#Write 3d results
file = File('laplacian.pvd')
phi = comp_phi(mesh, g_phi)
WW = VectorFunctionSpace(mesh, "CG", 3, dim=3)
projected = Function(WW, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#bilinear form for linearization
vv = Function(V, name='grad solution')
vv.sub(0).interpolate(g_phi)
vv.sub(1).interpolate(qq)
g_phi,qq = split(vv)

a = inner(p(g_phi) * g_phi[:,0].dx(0) + q(g_phi) * g_phi[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += 10 * inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx #rot stabilization
a += inner(g_psi[:,0].dx(1) - g_psi[:,1].dx(0), qq) * dx + inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), r) * dx #for mixed form

# Solving with Newton method
solve(a == 0, vv, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})
g_phi = vv.sub(0)
qq = vv.sub(1)

#Compute phi
phi = comp_phi(mesh, g_phi)
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file = File('res_newton.pvd')
file.write(projected)

WW = FunctionSpace(mesh, 'CG', 2)
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

#Shows \Omega'
WW = FunctionSpace(mesh, 'CG', 4)
aux = Function(WW)
aux.interpolate(sign(3 - inner(g_phi[:,0], g_phi[:,0])))# - 1e-1))
file = File('test.pvd')
file.write(aux)
