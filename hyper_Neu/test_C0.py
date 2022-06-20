#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
import numpy as np
parameters["form_compiler"].update({"optimize": True, "cpp_optimize": True, "representation":"uflacs", "quadrature_degree": 3})

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
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#  Dirichlet boundary conditions
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_ref = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))
g = as_vector((inner(phi_ref.dx(0),phi_ref.dx(0)), inner(phi_ref.dx(1),phi_ref.dx(1)), 0))

#initial guess
phi = Function(V, name='solution')
phi_t = TrialFunction(V)
psi = TestFunction(V)
#bcs = [DirichletBC(V, phi_ref, 11), DirichletBC(V, phi_ref, 5), DirichletBC(V, phi_ref, 6), DirichletBC(V, phi_ref, 8)]
bcs = [DirichletBC(V, phi_ref, 1), DirichletBC(V, phi_ref, 2)]
L = inner(Constant((0,0,0)), psi) * dx
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#Write 3d results
U = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
file = File('laplacian.pvd')
projected = Function(U, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)
#sys.exit()

#Writing our problem now
#phi.interpolate(phi_D)

#bilinear form for linearization
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi)) 
a = Gamma * inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx
#pen_term for new BC
h = CellDiameter(mesh)
pen = 1e1 #1e1
B_t = as_vector((inner(phi.dx(0), phi_t.dx(0)), inner(phi.dx(1), phi_t.dx(1)), inner(phi.dx(1), phi_t.dx(0))))
B = as_vector((inner(phi.dx(0), psi.dx(0)), inner(phi.dx(1), psi.dx(1)), inner(phi.dx(1), psi.dx(0))))
pen_term = pen * inner(B_t, B) * (ds(1)+ds(2)) #(ds(5)+ds(11))
L = pen * inner(g, B) * (ds(2)+ds(1)) #(ds(5)+ds(11))
a += pen_term
#pen continuity of derivative
h_avg = 0.5 * (h('+')+h('-'))
n = FacetNormal(mesh)
pen_cont = pen/h_avg**2 * inner(jump(grad(phi_t)), jump(grad(psi))) * dS - inner(avg( Gamma*(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1)) ), jump(grad(psi), n))*dS - inner(jump(grad(phi_t), n), avg( Gamma*(p(phi) * psi.dx(0).dx(0) + q(phi)*psi.dx(1).dx(1)) ))*dS
a += pen_cont

#BC to prevent rigid body rotation
#bcs = [DirichletBC(V, Constant((0,0,0)), 1), DirichletBC(V.sub(2), Constant(0), 2), DirichletBC(V.sub(2), Constant(0), 3), DirichletBC(V.sub(0), Constant(0), 4), DirichletBC(V, phi_ref, 6), DirichletBC(V, phi_ref, 8)]
#bcs = DirichletBC(V, phi_ref, 1)

# Picard iteration
tol = 1e-5 #1e-9
maxiter = 50
phi_old = Function(V) #for iterations
file = File('res.pvd')
for iter in range(maxiter):
  #linear solve
  A = assemble(a,bcs=bcs)
  b = assemble(L,bcs=bcs)
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  #checking translation
  blocked = phi.at((0,0))
  #assert np.linalg.norm(blocked) < 0.05 * abs(phi.vector()[:].max()) #checking that the disp at the origin is blocked
  #output surface
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
  
#Write 3d results
file = File('hyper.pvd')
x = SpatialCoordinate(mesh)
projected = Function(V, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Write 2d result
file_bis = File('flat.pvd')
file_bis.write(phi)

sys.exit()

