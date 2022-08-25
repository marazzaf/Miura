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
L = 2
H = 2
size_ref = 50
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
mesh = Mesh('mesh.msh')

# Define function space
V = TensorFunctionSpace(mesh, "CG", 1, shape=(3,2))
PETSc.Sys.Print('Nb dof: %i' % V.dim())
W = VectorFunctionSpace(mesh, "CG", 1, dim=3)
projected = Function(W, name='surface')
WW = FunctionSpace(mesh, 'CG', 2)

#Ref solution
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)

# Boundary conditions
a = 2
b = 4/3
phi_ref = as_vector((x[0], x[1], (x[0]-1)**2/a**2 - (x[1]-1)**2/b**2+x[0]+x[1]))
G_x = phi_ref.dx(0)
n_G_x_2 = inner(G_x, G_x)
n_G_y = sqrt(4/(4-n_G_x_2))
#aux = Function(WW)
#aux.interpolate(n_G_y)
#file = File('test.pvd')
#file.write(aux)
G_y = n_G_y * phi_ref.dx(1) / sqrt(inner(phi_ref.dx(1), phi_ref.dx(1)))
G = as_tensor((G_x, G_y)).T

#initial guess
#solve laplace equation on the domain
g_phi = Function(V, name='grad solution')
g_phi_t = TrialFunction(V)
g_psi = TestFunction(V)
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx #weak laplace
L = Constant(0) * g_psi[0,0] * dx

#Dirichlet BC
bcs = [DirichletBC(V, G, 2)]
#bcs = [DirichletBC(V, G, 1), DirichletBC(V, G, 2), DirichletBC(V, G, 3), DirichletBC(V, G, 4)]

#solving
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(A, g_phi, b, solver_parameters={'direct_solver': 'mumps'})
#solve(A, phi, b, solver_parameters={'ksp_type': 'cg','pc_type': 'bjacobi', 'ksp_rtol': 1e-5})
PETSc.Sys.Print('Laplace equation ok')

#Write 3d results
file = File('laplacian.pvd')
phi = comp_phi(mesh, g_phi)
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)
#sys.exit()

#bilinear form for linearization
a = inner(p(g_phi) * g_phi[:,0].dx(0) + q(g_phi) * g_phi[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx

# Solving with Newton method
#try:
solve(a == 0, g_phi, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25}) #25})

#Compute phi
#except exceptions.ConvergenceError:
phi = comp_phi(mesh, g_phi)
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file = File('phi.pvd')
file.write(projected)
  
#Try to restrict phi in another output to Omega'.
#This way, we'll only have what can be constructed.

u = Function(WW, name='u')
u.interpolate(inner(g_phi[:,0], g_phi[:,1]))
file = File('u.pvd')
file.write(u)
err = errornorm(Constant(0), u, 'l2')
PETSc.Sys.Print('L2 error in u: %.3e' % err)

v = Function(WW, name='v')
aux = 1 - 0.25 * inner(g_phi[:,0], g_phi[:,0])
v.interpolate(ln(aux * inner(g_phi[:,1], g_phi[:,1])))
#aux = v * ds(2)
#print(assemble(aux))
#sys.exit()
file = File('v.pvd')
file.write(v)
err = errornorm(Constant(1), v, 'l2')
PETSc.Sys.Print('L2 error in v: %.3e' % err)

aux = Function(W)
aux.interpolate(g_phi[:,0])
file = File('phi_x.pvd')
file.write(aux)
file = File('phi_y.pvd')
aux.interpolate(g_phi[:,1])
file.write(aux)

x = Function(WW, name='phi_x')
x.interpolate(inner(g_phi[:,0], g_phi[:,0]))
y = Function(WW, name='phi_y')
y.interpolate(inner(g_phi[:,1], g_phi[:,1]))
PETSc.Sys.Print('Min norm of phi_y squared: %.5e' % min(y.vector().array()))
