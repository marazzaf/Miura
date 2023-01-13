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
  truc = conditional(gt(sq, Constant(4)), Constant(1), aux)
  truc2 = conditional(lt(sq, Constant(1)), Constant(4), truc)
  return truc2

# Create mesh
L = 2
H = 2
#size_ref = 100
#mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
mesh = Mesh('mesh_1.msh')

# Define function space
#W = TensorFunctionSpace(mesh, "CG", 2, shape=(3,2))
#Q = VectorFunctionSpace(mesh, "CG", 1, dim=3)
#V = W * Q
V = TensorFunctionSpace(mesh, "CG", 1, shape=(3,2))
PETSc.Sys.Print('Nb dof: %i' % V.dim())

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
aux = phi_ref.dx(1) - inner(G_x, phi_ref.dx(1)) * G_x / inner(G_x, G_x)
G_y = n_G_y * aux / sqrt(inner(aux, aux))
G = as_tensor((G_x, G_y)).T

#initial guess
#solve laplace equation on the domain
#g_phi_t,q_t = TrialFunctions(V)
#g_psi,r = TestFunctions(V)
g_phi_t = TrialFunction(V)
g_psi = TestFunction(V)
pen = 1e1
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx #+ pen * inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx #laplace in weak form
#laplace += inner(g_psi[:,0].dx(1) - g_psi[:,1].dx(0), q_t) * dx + inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), r) * dx
L = Constant(0) * g_psi[0,0] * dx

#Dirichlet BC
#bcs = [DirichletBC(V.sub(0), G, 2)]
bcs = [DirichletBC(V, G, 2)]


#solving
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
#v = Function(V, name='grad solution')
g_phi = Function(V, name='grad solution')
#solve(A, v, b, solver_parameters={'direct_solver': 'mumps'})
solve(A, g_phi, b, solver_parameters={'direct_solver': 'mumps'})
#g_phi,qq = v.split()
PETSc.Sys.Print('Laplace equation ok')

#Write 3d results
file = File('laplacian.pvd')
phi = comp_phi(mesh, g_phi)
WW = VectorFunctionSpace(mesh, "CG", 3, dim=3)
projected = Function(WW, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Check if curl of the solution inside is zero
WW = FunctionSpace(mesh, 'CG', 1)
truc = g_phi[:,1].dx(0) - g_phi[:,0].dx(1) 
aux = Function(WW)
aux.interpolate(inner(truc, truc))
file = File('curl.pvd')
file.write(aux)
sys.exit()

#bilinear form for linearization
v = Function(V, name='grad solution')
g_phi,qq = split(v)

a = inner(p(g_phi) * g_phi[:,0].dx(0) + q(g_phi) * g_phi[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += pen * inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx #rot stabilization
a += inner(g_psi[:,0].dx(1) - g_psi[:,1].dx(0), qq) * dx + inner(g_phi[:,0].dx(1) - g_phi[:,1].dx(0), r) * dx #for mixed form

# Solving with Newton method
solve(a == 0, v, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25}) #25})
g_phi,qq = v.split()

#Compute phi
#except exceptions.ConvergenceError:
phi = comp_phi(mesh, g_phi)
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file = File('phi.pvd')
file.write(projected)
  
#Try to restrict phi in another output to Omega'.
#This way, we'll only have what can be constructed.
WW = FunctionSpace(mesh, 'CG', 4)
aux = Function(WW)
aux.interpolate(sign(inner(g_phi[:,1], g_phi[:,1]) - 1))
file = File('test.pvd')
file.write(aux)

u = Function(WW, name='u')
u.interpolate(inner(g_phi[:,0], g_phi[:,1]))
file = File('u.pvd')
file.write(u)
err = errornorm(Constant(0), u, 'l2')
PETSc.Sys.Print('L2 error in u: %.3e' % err)

v = Function(WW, name='v')
aux = 1 - 0.25 * inner(g_phi[:,0], g_phi[:,0])
v.interpolate(ln(aux * inner(g_phi[:,1], g_phi[:,1])))
file = File('v.pvd')
file.write(v)
err = errornorm(Constant(1), v, 'l2')
PETSc.Sys.Print('L2 error in v: %.3e' % err)

x = Function(WW, name='phi_x')
x.interpolate(inner(g_phi[:,0], g_phi[:,0]))
file = File('phi_x.pvd')
file.write(x)
y = Function(WW, name='phi_y')
y.interpolate(inner(g_phi[:,1], g_phi[:,1]))
PETSc.Sys.Print('Min norm of phi_y squared: %.5e' % min(y.vector().array()))
file = File('phi_y.pvd')
file.write(y)
