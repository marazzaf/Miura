#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
import numpy as np
sys.path.append('..')
from comp_phi import comp_phi
from ufl import sign

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
L = 0.75
H = 2*pi
size_ref = 25 #50
mesh = PeriodicRectangleMesh(size_ref, 6*size_ref, L, H, diagonal='crossed', direction='y')

# Define function space
W = TensorFunctionSpace(mesh, "CG", 1, shape=(3,2))
Q = VectorFunctionSpace(mesh, "CG", 1, dim=3)
V = W * Q
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#Ref solution
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)

## Boundary conditions
theta_s = pi/2
e_r = as_vector((cos(x[1])*sin(theta_s), sin(x[1])*sin(theta_s), cos(theta_s)))
e_theta = as_vector((cos(x[1])*cos(theta_s), sin(x[1])*cos(theta_s), -sin(theta_s)))
e_phi = as_vector((-sin(x[1]), cos(x[1]), 0))
rho_m = sqrt(4/3)
G_x = ((2*sqrt(1 - 0.5/rho_m)-1)/L * x[0] + 1) * e_r
n_G_x_2 = inner(G_x, G_x)
n_G_y = sqrt(4/(4-n_G_x_2))
G_y = n_G_y * e_phi
G = as_tensor((G_x, G_y)).T

#initial guess
#solve laplace equation on the domain
v = Function(V, name='grad solution')
g_phi_t,q_t = TrialFunctions(V)
g_psi,r = TestFunctions(V)
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx + 10 * inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx #laplace in weak form
laplace += inner(g_psi[:,0].dx(1) - g_psi[:,1].dx(0), q_t) * dx + inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), r) * dx
zero = Constant(0) * g_psi[0,0] * dx

#Dirichlet BC
bcs = [DirichletBC(V.sub(0), G, 1), DirichletBC(V.sub(0), G, 2)] #, DirichletBC(V, G, 3), DirichletBC(V, G, 4)]

#solving
A = assemble(laplace, bcs=bcs)
b = assemble(zero, bcs=bcs)
solve(A, v, b, solver_parameters={'direct_solver': 'mumps'})
g_phi,qq = v.split()
PETSc.Sys.Print('Laplace equation ok')

#Write 3d results
file = File('laplacian.pvd')
phi = comp_phi(mesh, g_phi)
WW = VectorFunctionSpace(mesh, "CG", 3, dim=3)
projected = Function(WW, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#bilinear form for linearization
v = Function(V, name='grad solution')
g_phi,qq = split(v)

a = inner(p(g_phi) * g_phi[:,0].dx(0) + q(g_phi) * g_phi[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += 10 * inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx #rot stabilization
a += inner(g_psi[:,0].dx(1) - g_psi[:,1].dx(0), qq) * dx + inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), r) * dx #for mixed form

# Solving with Newton method
#try:
solve(a == 0, v, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25}) #25})
g_phi,qq = v.split()

#Compute phi
#except exceptions.ConvergenceError:
phi = comp_phi(mesh, g_phi)
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file = File('phi.pvd')
file.write(projected)

#Check if curl of the solution inside is zero
WW = FunctionSpace(mesh, 'CG', 1)
truc = g_phi[:,1].dx(0) - g_phi[:,0].dx(1) 
aux = Function(WW)
aux.interpolate(inner(truc, truc))
file = File('curl.pvd')
file.write(aux)

#Try to restrict phi in another output to Omega'.
#This way, we'll only have what can be constructed.
aux = Function(WW)
aux.interpolate(sign(inner(g_phi[:,1], g_phi[:,1]) - 1))
file = File('test.pvd')
file.write(aux)
#sys.exit()
#
#u = Function(WW, name='u')
#u.interpolate(inner(g_phi[:,0], g_phi[:,1]))
#file = File('u.pvd')
#file.write(u)
#err = errornorm(Constant(0), u, 'l2')
#PETSc.Sys.Print('L2 error in u: %.3e' % err)
#
#v = Function(WW, name='v')
#aux = 1 - 0.25 * inner(g_phi[:,0], g_phi[:,0])
#v.interpolate(ln(aux * inner(g_phi[:,1], g_phi[:,1])))
##aux = v * ds(2)
##print(assemble(aux))
##sys.exit()
#file = File('v.pvd')
#file.write(v)
#err = errornorm(Constant(1), v, 'l2')
#PETSc.Sys.Print('L2 error in v: %.3e' % err)
#
#aux = Function(W)
#aux.interpolate(g_phi[:,0])
#file = File('phi_x.pvd')
#file.write(aux)
#file = File('phi_y.pvd')
#aux.interpolate(g_phi[:,1])
#file.write(aux)
#file = File('normal.pvd')
#aux.interpolate(cross(g_phi[:,0], g_phi[:,1]))
#file.write(aux)
##sys.exit()

x = Function(WW, name='phi_x')
x.interpolate(inner(g_phi[:,0], g_phi[:,0]))
file = File('phi_x.pvd')
file.write(x)
y = Function(WW, name='phi_y')
y.interpolate(inner(g_phi[:,1], g_phi[:,1]))
PETSc.Sys.Print('Min norm of phi_y squared: %.5e' % min(y.vector().array()))
file = File('phi_y.pvd')
file.write(y)
sys.exit()

#Verif aux eq
WW = FunctionSpace(mesh, 'DG', 0)
test_1 = p(g_phi)*u.dx(0) + 2*v.dx(1)
#test_1 = 2*p(g_phi)*u.dx(0) + v.dx(1)
aux = Function(WW)
aux.interpolate(test_1)
file = File('test_1.pvd')
file.write(aux)
test_2 = q(g_phi)*u.dx(1) - 2*v.dx(0)
#test_2 = 2*p(g_phi)*u.dx(1) - v.dx(0)
aux.interpolate(test_2)
file = File('test_2.pvd')
file.write(aux)

#Write 3d results
mesh = RectangleMesh(size_ref, 6*size_ref, L*0.999, H*0.999, diagonal='crossed')
W = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
X = interpolate(mesh.coordinates, VectorFunctionSpace(mesh, 'CG', 1))

#gives values from phi
def func(data):
  res = np.zeros((len(data),3))
  for i,dat in enumerate(data):
    res[i,:] = phi(dat)
  return res

# Use the external data function to interpolate the values of f.
phi_bis = Function(W)
phi_bis.dat.data[:] = func(X.dat.data_ro)

#interpolation on new mesh
projected = Function(W, name='surface')
file = File('new.pvd')
x = SpatialCoordinate(mesh)
#projected.interpolate(as_vector((x[0], x[1], 0)))
projected.interpolate(phi_bis - as_vector((x[0], x[1], 0)))
file.write(projected)
