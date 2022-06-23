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

# Create mesh and define function space
LL = 2 #length of rectangle
H = 1 #height of rectangle
mesh= Mesh('mesh_1.msh')
size_ref = 1
V = VectorFunctionSpace(mesh, "CG", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
W = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)

# Boundary conditions
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)
aux = x[0]+x[1] #x[0]**2 + x[1]**2
g1 = conditional(gt(aux, Constant(3)), Constant(3), aux)
g2 = 4 / (4 - g1)
g = as_vector((g1, g2, 0)) #g1))

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations
phi.project(as_vector((x[0],-x[1],x[0])))
#Defining the bilinear forms
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)

#penalty to impose new BC
h = CellDiameter(mesh)
pen = 1e1
B_t = as_vector((inner(phi.dx(0), phi_t.dx(0)), inner(phi.dx(1), phi_t.dx(1)), inner(phi.dx(1), phi_t.dx(0))))
B = as_vector((inner(phi.dx(0), psi.dx(0)), inner(phi.dx(1), psi.dx(1)), inner(phi.dx(1), psi.dx(0))))
pen_term = pen * inner(B_t, B) * ds #(ds(5)+ds(11)+ds(8)+ds(6))
L = pen * inner(g, B) * ds #(ds(5)+ds(11)+ds(8)+ds(6))

##penalty for Neumann BC
#pen_term = pen * inner(dot(grad(phi_t),n), dot(grad(psi),n)) * ds #(ds(5)+ds(11)+ds(8)+ds(6))
#L = pen * inner(g, dot(grad(psi),n)) * ds #(ds(5)+ds(11)+ds(8)+ds(6))

#penalty terms so solution can't move
#for translation
pen_disp = pen/h**4 * inner(phi_t,psi) * ds(1)
#for directions
tau_4 = Constant((1,0,0))
pen_rot = pen/h**4 * inner(phi_t,tau_4) * inner(psi,tau_4)  * ds(4) #e_z blocked
tau_3 = Constant((0,0,1))
pen_rot += pen/h**4 * inner(phi_t,tau_3) * inner(psi,tau_3)  * ds(3) #e_x blocked
tau_2 = Constant((0,0,1))
pen_rot += pen/h**4 * inner(phi_t,tau_2) * inner(psi,tau_2)  * ds(2) #e_y blocked

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
#laplace = inner(div(grad(phi_t)), div(grad(psi))) * dx #test
A = assemble(laplace+pen_term+pen_disp+pen_rot)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

file_bis = File('laplacian.pvd')
proj = project(phi - as_vector((x[0],x[1],0)), W, name='laplacian')
file_bis.write(proj)
#sys.exit()

#Bilinear form
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi)) 
a = Gamma * inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx
a += pen_term + pen_disp +pen_rot

file = File('res.pvd')

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
for iter in range(maxiter):
  #linear solve
  A = assemble(a)
  b = assemble(L)
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
  
  #convergence test 
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  #output
  projected = Function(W, name='surface')
  projected.interpolate(phi - as_vector((x[0], x[1], 0)))
  file.write(projected)

  #assert on disp and directions?
  projected.interpolate(phi - 1e-5*as_vector((x[0], x[1], 0)))
  blocked = projected.at((0,0))
  assert np.linalg.norm(blocked) < 0.05 * abs(projected.vector()[:].max()) #checking that the disp at the origin is blocked
  blocked = projected.at((LL-1e-5,0))[2]
  assert abs(blocked) < 0.05 * abs(projected.vector()[:].max())
  blocked = projected.at((LL-1e-5,H-1e-5))[2]
  assert abs(blocked) < 0.05 * abs(projected.vector()[:].max())
  blocked = projected.at((0,H-1e-5))[0]
  assert abs(blocked) < 0.05 * abs(projected.vector()[:].max())
  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Write 2d results
flat = File('flat_%i.pvd' % size_ref)
proj = project(phi, W, name='flat')
flat.write(proj)
  
#Write 3d results
file = File('new_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
projected = Function(W, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Test is inequalities are true
file_bis = File('verif_x.pvd')
proj = project(inner(phi.dx(0),phi.dx(0)), UU, name='test phi_x')
file_bis.write(proj)
file_ter = File('verif_y.pvd')
proj = project(inner(phi.dx(1),phi.dx(1)), UU, name='test phi_y')
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj = project(inner(phi.dx(0),phi.dx(1)), UU, name='test PS')
file_4.write(proj)
