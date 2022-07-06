#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
sys.path.append('../hyper_Neu')
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

# Create mesh and define function space
L = 2 #length of rectangle
H = 3/sqrt(2)*L #height of rectangle
size_ref = 25 #50 #degub: 2
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
V = TensorFunctionSpace(mesh, "CG", 2, shape=(3,2))
PETSc.Sys.Print('Nb dof: %i' % V.dim())

# Boundary conditions
x = SpatialCoordinate(mesh)
phi_D1 = as_vector((x[0], 2/sqrt(3)*x[1], 0))

#other part of BC
alpha = pi/2 #pi/4
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
phi_D2 = (1-x[0]/L)*BpA + (1-x[1]/H)*BpC + OBp

#test
phi_ref = x[0]*x[1]/L/H * (OBp - OC - Constant((0, H, 0))) + Constant((1, 0, 0)) * x[0] + Constant((0, 1, 0)) * x[1]

# Creating function to store solution
g_phi = Function(V, name='solution')
phi = as_vector((x[0], x[1], 0))
g_phi.interpolate(grad(phi))
g_phi_old = Function(V) #for iterations

#Defining the bilinear forms
#bilinear form for linearization
g_phi_t = TrialFunction(V)
g_psi = TestFunction(V)
a = inner(p(g_phi) * g_phi_t[:,0].dx(0) + q(g_phi) * g_phi_t[:,1].dx(1),  p(g_phi) * g_psi[:,0].dx(0) + q(g_phi) * g_psi[:,1].dx(1)) * dx
a += inner(g_phi_t[:,0].dx(1) - g_phi_t[:,1].dx(0), g_psi[:,0].dx(1) - g_psi[:,1].dx(0)) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1

#Dirichlet BC other form
#lhs
pen_term = 0.5*pen/h**4 * inner(g_phi[:,1], g_phi_t[:,0]) * inner(g_phi[:,1], g_psi[:,0]) * ds + 0.5*pen/h**4 * inner(g_phi[:,0], g_phi_t[:,1]) * inner(g_phi[:,0], g_psi[:,1]) * ds
pen_term += pen/h**4 * inner(g_phi[:,0], g_phi_t[:,0]) * inner(g_phi[:,0], g_psi[:,0]) * ds + pen/h**4 * inner(g_phi[:,1], g_phi_t[:,1]) * inner(g_phi[:,1], g_psi[:,1]) * ds
pen_term += 0.5*pen/h**4 * inner(cross(g_phi[:,0], g_phi_t[:,1]), cross(g_phi[:,0], g_psi[:,1])) * ds + 0.5*pen/h**4 * inner(cross(g_phi[:,1], g_phi_t[:,0]), cross(g_phi[:,1], g_psi[:,0])) * ds

#rhs
L = pen/h**4 * inner(grad(phi_ref)[:,0], grad(phi_ref)[:,0]) * inner(g_phi[:,0], g_psi[:,0]) * ds + pen/h**4 * inner(grad(phi_ref)[:,1], grad(phi_ref)[:,1]) * inner(g_phi[:,1], g_psi[:,1]) * ds
L += 0.5*pen/h**4 * inner(cross(g_phi[:,0], grad(phi_ref)[:,1]), cross(g_phi[:,0], g_psi[:,1])) * ds + 0.5*pen/h**4 * inner(cross(g_phi[:,1], grad(phi_ref)[:,0]), cross(g_phi[:,1], g_psi[:,0])) * ds

#Computing initial guess
laplace = inner(grad(g_phi_t), grad(g_psi)) * dx #laplace in weak form
#laplace = inner(div(grad(phi_t)), div(grad(psi))) * dx #test
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, g_phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#test
eps = sqrt(assemble(inner(grad(g_phi-g_phi_old), grad(g_phi-g_phi_old))*dx)) # check increment size as convergence test
PETSc.Sys.Print('Before computation  H1 seminorm of delta: {:10.2e}'.format(eps))

#Write 3d results
file = File('laplacian.pvd')
phi = comp_phi(mesh, g_phi)
W = VectorFunctionSpace(mesh, "CG", 3, dim=3)
projected = Function(W, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
file = File('res.pvd')
for iter in range(maxiter):
  #linear solve
  A = assemble(a + pen_term)
  b = assemble(L)
  solve(A, g_phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
  
  #convergence test 
  eps = sqrt(assemble(inner(grad(g_phi-g_phi_old), grad(g_phi-g_phi_old))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H1 seminorm of delta: {:10.2e}'.format(iter+1, eps))

  #Compute phi
  phi = comp_phi(mesh, g_phi)
  projected.interpolate(phi - as_vector((x[0], x[1], 0)))
  file.write(projected)
  
  if eps < tol:
    break
  g_phi_old.assign(g_phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Write 2d results
flat = File('flat_%i.pvd' % size_ref)
W = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
proj = project(phi, W, name='flat')
flat.write(proj)
  
#Write 3d results
file = File('new_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
projected = Function(W, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Test is inequalities are true
W = FunctionSpace(mesh, "CG", 2)
file_bis = File('verif_x.pvd')
proj = Function(W)
proj.interpolate(inner(g_phi[:,0],g_phi[:,0]))
file_bis.write(proj)
file_ter = File('verif_y.pvd')
proj.interpolate(inner(g_phi[:,1],g_phi[:,1]))
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj.interpolate(inner(g_phi[:,0],g_phi[:,1]))
file_4.write(proj)
