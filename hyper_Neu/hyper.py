#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys

# the coefficient functions
def p(phi):
  aux = 1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))
  return interpolate(conditional(lt(aux, Constant(1)), Constant(100), aux), UU) #update maybe?

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1)) #update maybe?

# Size for the domain
theta = pi/2 #pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle
l = sin(theta/2)*L

#Creating mesh
mesh = Mesh('mesh.msh')
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
VV = FunctionSpace(mesh, 'CG', 4)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
UU = FunctionSpace(mesh, 'CG', 4)

#  Ref solution
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_ref = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))
g = as_vector((inner(phi_ref.dx(0),phi_ref.dx(0)), inner(phi_ref.dx(1),phi_ref.dx(1)), 0))

#initial guess
#solve laplace equation on the domain
phi = Function(V, name='solution')
phi.project(phi_ref) #for now
phi_t = TrialFunction(V)
psi = TestFunction(V)
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form

#penalty term for new BC
h = CellDiameter(mesh)
pen = 1e1 #1e1
B_t = as_vector((inner(phi.dx(0), phi_t.dx(0)), inner(phi.dx(1), phi_t.dx(1)), 0.5*(inner(phi.dx(0), phi_t.dx(1)) + inner(phi.dx(1), phi_t.dx(0)))))
B = as_vector((inner(phi.dx(0), psi.dx(0)), inner(phi.dx(1), psi.dx(1)), 0.5*(inner(phi.dx(0), psi.dx(1)) + inner(phi.dx(1), psi.dx(0)))))
pen_term = pen * inner(B_t, B) * ds
L = pen * inner(g, B) * ds

#penalty term to remove the invariance
#Define the surface of the boundary
hF = 0.5 * (h('+') + h('-'))
pen_disp = pen/hF**4 * inner(phi_t,psi) * ds(1)
#pen_disp = pen * inner(phi_t * dx, psi * dx)
#for directions
tau_1 = Constant((0,1,0))
pen_rot = pen * inner(phi_t,tau_1) * inner(psi,tau_1)  * ds(2)
tau_2 = Constant((-1,0,0))
pen_rot += pen * inner(phi_t,tau_1) * inner(psi,tau_1)  * ds(3)
tau_3 = Constant((0,0,-1))
pen_rot += pen * inner(phi_t,tau_1) * inner(psi,tau_1)  * ds(4)

#solving
A = assemble(laplace+pen_term+pen_disp+pen_rot)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
#solve(A, phi, b, solver_parameters={'ksp_type': 'cg','pc_type': 'bjacobi', 'ksp_rtol': 1e-5})
PETSc.Sys.Print('Laplace equation ok')

##Write 3d results
#U = VectorFunctionSpace(mesh, 'CG', 4, dim=3)
#file = File('laplacian.pvd')
#x = SpatialCoordinate(mesh)
#projected = Function(U, name='surface')
#projected.interpolate(phi - as_vector((x[0], x[1], 0)))
#file.write(projected)
sys.exit()

#Writing our problem now
#bilinear form for linearization
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi)) 
a = Gamma * inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
a += pen_term

# Solving with Newton method
solve(a == 0, phi, solver_parameters={'snes_monitor': None}) 

#Computing error
X = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
projected = interpolate(div(grad(phi)), X)
ref = interpolate(div(grad(phi_D)), X)
err = sqrt(assemble(inner(div(grad(phi-phi_D)), div(grad(phi-phi_D)))*dx))
#PETSc.Sys.Print('Error: %.3e' % err)

#For projection
U = VectorFunctionSpace(mesh, 'CG', 4, dim=3)

#Tests if inequalities are true
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

