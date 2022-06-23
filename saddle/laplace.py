#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
import numpy as np

# Create mesh and define function space
LL = 1 #length of rectangle
H = 1 #height of rectangle
mesh= Mesh('mesh_1.msh')
size_ref = 1
V = VectorFunctionSpace(mesh, "CG", 4, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

# Boundary conditions
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)
#aux = x[0]**2 + x[1]**2
#g1 = conditional(gt(aux, Constant(3)), Constant(3), aux)
#g2 = 4 / (4 - g1)
#g = as_vector((g1, g2, 0))

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations
phi.interpolate(as_vector((x[0],-x[1],0))) #Initial guess...
#phi_old.interpolate(as_vector((x[0],sqrt(2)*x[1],0)))
#phi.interpolate(as_vector((x[0],sqrt(2)*x[1],0)))
phi_ref = as_vector((x[0], sqrt(2)*x[1], 0))
g = Constant((1, 2, 0))
N = cross(phi.dx(0), phi.dx(1)) / norm(cross(phi.dx(0), phi.dx(1)))
phi_D = 0

file = File('test.pvd')
file.write(interpolate(phi_ref, V))

#Defining the bilinear forms
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)

#penalty to impose new BC
h = CellDiameter(mesh)
pen = 1e1
#B_t = as_vector((inner(phi_ref.dx(0), phi_t.dx(0)), inner(phi_ref.dx(1), phi_t.dx(1)), inner(phi_ref.dx(1), phi_t.dx(0))))
#B = as_vector((inner(phi_ref.dx(0), psi.dx(0)), inner(phi_ref.dx(1), psi.dx(1)), inner(phi_ref.dx(1), psi.dx(0))))
B_t = as_vector((inner(phi.dx(0), phi_t.dx(0)), inner(phi.dx(1), phi_t.dx(1)), 0))
B = as_vector((inner(phi.dx(0), psi.dx(0)), inner(phi.dx(1), psi.dx(1)), 0))
pen_term = pen * inner(B_t, B) * ds + pen/h**2 * inner(phi_t, N) * inner(psi, N) * ds
L = pen * inner(g, B) * ds # + pen/h**2 * phi_D * inner(psi, N) * ds

##penalty for Neumann BC
#pen_term = pen * inner(dot(grad(phi_t),n), dot(grad(psi),n)) * ds #(ds(5)+ds(11)+ds(8)+ds(6))
#L = pen * inner(dot(grad(phi_ref),n), dot(grad(psi),n)) * ds #(ds(5)+ds(11)+ds(8)+ds(6))
#L = inner(dot(grad(phi_ref),n), psi) * ds

#Dirichlet BC
#pen_term = pen/h**2 * inner(phi_t, psi) * ds
#L = pen/h**2 * inner(phi_ref, psi) * ds
N = cross(phi_ref.dx(0), phi_ref.dx(1)) / norm(cross(phi_ref.dx(0), phi_ref.dx(1)))
pen_term = pen/h**2 * inner(phi_t, phi_ref.dx(0)) * inner(psi, phi_ref.dx(0)) * ds + pen/h**2 * inner(phi_t, phi_ref.dx(1)) * inner(psi, phi_ref.dx(1)) * ds + pen/h**2 * inner(phi_t, N) * inner(psi, N) * ds
L = pen/h**2 * inner(phi_ref, phi_ref.dx(0)) * inner(psi, phi_ref.dx(0)) * ds + pen/h**2 * inner(phi_ref, phi_ref.dx(1)) * inner(psi, phi_ref.dx(1)) * ds + pen/h**2 * inner(phi_ref, N) * inner(psi, N) * ds

#Bilinear form
#laplace = inner(div(grad(phi_t)), div(grad(psi))) * dx #laplace in weak form
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
#laplace = inner(div(grad(phi_t)), div(grad(psi))) * dx #test
a = laplace + pen_term - inner(dot(grad(phi_t),n), psi) * ds

#test = assemble(action(a, phi) - L).vector().sum()
#print(test)
#sys.exit()

#BC to have uniqueness
bcs = [DirichletBC(V, Constant((0,0,0)), 1), DirichletBC(V.sub(0), phi_ref[0], 4), DirichletBC(V.sub(2), phi_ref[2], 3), DirichletBC(V.sub(2), phi_ref[2], 2)]
#bcs = [DirichletBC(V, Constant((0,0,0)), 1), DirichletBC(V.sub(0), Constant(0), 4), DirichletBC(V.sub(2), Constant(0), 3), DirichletBC(V.sub(2), Constant(0), 2)]

#pen terms to have uniqueness
pen_disp = pen/h**4 * inner(phi_t,psi) * ds(1)
#for directions
tau_4 = Constant((1,0,0))
pen_rot = pen/h**4 * inner(phi_t,tau_4) * inner(psi,tau_4)  * ds(4) #e_z blocked
tau_3 = Constant((0,0,1))
pen_rot += pen/h**4 * inner(phi_t,tau_3) * inner(psi,tau_3)  * ds(3) #e_x blocked
tau_2 = Constant((0,0,1))
pen_rot += pen/h**4 * inner(phi_t,tau_2) * inner(psi,tau_2)  * ds(2) #e_y blocked

#a += pen_disp + pern_rot

file = File('res_laplace.pvd')

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
for iter in range(maxiter):
  #linear solve
  A = assemble(a, bcs=bcs)
  b = assemble(L, bcs=bcs)
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate

  test = assemble(action(a, phi_old) - L).vector().sum()
  print(test)
  
  #convergence test 
  eps = sqrt(assemble(inner(grad(phi-phi_old), grad(phi-phi_old))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H1 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  #output
  projected = Function(V, name='surface')
  projected.interpolate(phi) # - as_vector((x[0], x[1], 0)))
  file.write(projected)
  #sys.exit()

  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

sys.exit()

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
