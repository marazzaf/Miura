#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys
parameters["form_compiler"].update({"optimize": True, "cpp_optimize": True, "representation":"uflacs", "quadrature_degree": 3})

# the coefficient functions
def p(phi):
  aux = 1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))
  return conditional(lt(aux, Constant(1)), Constant(100), aux)

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

# Size for the domain
theta = pi/2 #pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle
l = sin(theta/2)*L

#Creating mesh
mesh = Mesh('rectangle.msh')
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#  Dirichlet boundary conditions
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

#initial guess
phi = Function(V, name='solution')
phi_t = TrialFunction(V)
psi = TestFunction(V)
#solve laplace equation on the domain
bcs = [DirichletBC(V, phi_D, 1), DirichletBC(V, phi_D, 2), DirichletBC(V, phi_D, 3), DirichletBC(V, phi_D, 4), DirichletBC(V, phi_D, 5)]
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
L = inner(Constant((0,0,0)), psi) * dx
A = assemble(laplace, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#Writing our problem now
phi.interpolate(phi_D)

#bilinear form for linearization
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx
h = CellDiameter(mesh)
h_avg = 0.5 * (h('+')+h('-'))
n = FacetNormal(mesh)
pen = 1e1
pen_term = pen/h_avg**2 * inner(jump(grad(phi_t)), jump(grad(psi))) * dS - inner(avg( p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1) ), jump(grad(psi), n))*dS - inner(jump(grad(phi_t), n), avg( p(phi) * psi.dx(0).dx(0) + q(phi)*psi.dx(1).dx(1) ))*dS
a += pen_term

#BC to prevent rigid body rotation
bc = [DirichletBC(V, phi_D, 5)]

#Forms to impose the BC
phi_x = phi_D.dx(0)
phi_y = phi_D.dx(1)
#n = cross(phi_D.dx(0), phi_D.dx(1))
#n /= sqrt(inner(n, n))
pen = 10
pen_term = pen/h**2 * dot(phi_t.dx(0), psi.dx(0)) * ds + pen/h**2 * dot(phi_t.dx(1), psi.dx(1)) * ds # + pen/h**2 * dot(phi_t, n) * dot(psi, n) * ds
a += pen_term
L = pen/h**2 * dot(phi.dx(0), psi.dx(0)) * ds + pen/h**2 * dot(phi.dx(1), psi.dx(1)) * ds # + pen/h**2 * dot(phi_D, n) * dot(psi, n) * ds
#pen_scal = 
#a += pen_scal

# Picard iteration
tol = 1e-5 #1e-9
maxiter = 50
phi_old = Function(V) #for iterations
for iter in range(maxiter):
  #linear solve
  A = assemble(a, bcs=bc)
  b = assemble(L, bcs=bc)
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
    
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))

  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Computing error
X = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
projected = interpolate(div(grad(phi)), X)
ref = interpolate(div(grad(phi_D)), X)
err = sqrt(assemble(inner(div(grad(phi-phi_D)), div(grad(phi-phi_D)))*dx))
PETSc.Sys.Print('Error: %.3e' % err)

#Write 3d results
file = File('hyper.pvd')
x = SpatialCoordinate(mesh)
projected = Function(U, name='surface')
projected.interpolate(phi - as_vector((x[0], x[1], 0)))
file.write(projected)

#Write 2d result
file_bis = File('flat.pvd')
#proj = project(phi, U, name='flat')
proj = Function(U, name='flat')
proj.interpolate(phi - 1.e-5*as_vector((x[0], x[1], 0)))
file_bis.write(proj)

sys.exit()

#Test if inequalities are true
file_bis = File('verif_x.pvd')
phi_x = interpolate(phi.dx(0), U)
proj = project(inner(phi_x,phi_x), UU, name='test phi_x')
file_bis.write(proj)
file_ter = File('verif_y.pvd')
phi_y = interpolate(phi.dx(1), U)
proj = project(inner(phi_y,phi_y), UU, name='test phi_y')
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj = project(inner(phi_x,phi_y), UU, name='test PS')
file_4.write(proj)
