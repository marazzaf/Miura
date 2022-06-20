#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys

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
L = 2 #length of rectangle
H = 3/sqrt(2)*L #height of rectangle
size_ref = 25 #50 #degub: 2
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
#mesh = UnitDiskMesh(size_ref)
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)
P = Function(UU)

# Boundary conditions
x = SpatialCoordinate(mesh)
phi_D1 = as_vector((x[0], 2/sqrt(3)*x[1], 0))

#file = File('BC_1.pvd')
#x = SpatialCoordinate(mesh)
#projected = Function(U, name='surface')
#projected.interpolate(phi_D1 - as_vector((x[0], x[1], 0)))
#file.write(projected)

#other part of BC
HH = 2/sqrt(3)*H
alpha = pi/2 #pi/2 #pi/4
l = H*L / sqrt(L*L + HH*HH)
sin_gamma = HH / sqrt(L*L+HH*HH) 
cos_gamma = L / sqrt(L*L+HH*HH) 
DB = l*as_vector((sin_gamma,cos_gamma,0))
DBp = l*sin(alpha)*Constant((0,0,1)) + cos(alpha) * DB
OA = as_vector((L, 0, 0))
AD = Constant((-sin_gamma*l,HH-cos_gamma*l,0))
OBp = OA + AD + DBp
BpC = -DBp - AD + Constant((-L, HH, 0))
BpA = BpC + Constant((L, -HH, 0))
phi_D2 = (1-x[0]/L)*BpC + (1-x[1]/H)*BpA + OBp

#file = File('BC_2.pvd')
#x = SpatialCoordinate(mesh)
#projected = Function(U, name='surface')
#projected.interpolate(phi_D2 - as_vector((x[0], x[1], 0)))
#file.write(projected)
#sys.exit()

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations

#Defining the bilinear forms
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1
#lhs
pen_term = pen/h**4 * inner(phi_t, psi) * ds
#rhs
#L = pen/h**4 * inner(phi_D1, psi) * ds
L = pen/h**4 * inner(phi_D1, psi) *(ds(1)+ds(3)) + pen/h**4 * inner(phi_D2, psi) *(ds(2)+ds(4))

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
#laplace = inner(div(grad(phi_t)), div(grad(psi))) * dx #test
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#Dirichlet pen term
pen_term = pen/h**4 * inner(phi_t, psi) * (ds(1)+ds(3)) 
L = pen/h**4 * inner(phi_D1, psi) * (ds(1)+ds(3)) 
##pen term for new BC
#pen = 1e1 #1e1
#B_t = as_vector((inner(phi.dx(0), phi_t.dx(0)), inner(phi.dx(1), phi_t.dx(1)), inner(phi.dx(1), phi_t.dx(0))))
#B = as_vector((inner(phi.dx(0), psi.dx(0)), inner(phi.dx(1), psi.dx(1)), inner(phi.dx(1), psi.dx(0))))
#g = as_vector((inner(phi_D2.dx(0),phi_D2.dx(0)), inner(phi_D2.dx(1),phi_D2.dx(1)), 0)) 
#pen_term += pen * inner(B_t, B) * (ds(2)+ds(4))
#L += pen * inner(g, B) * (ds(2)+ds(4))
pen_term += pen/h**4 * inner(phi_t, psi) * (ds(2)+ds(4)) 
L += pen/h**4 * inner(phi_D2, psi) * (ds(2)+ds(4)) 

#Bilinear form
Gamma = (p(phi) + q(phi)) / (p(phi)*p(phi) + q(phi)*q(phi)) 
a = Gamma * inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx
a += pen_term

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
  if eps < tol:
    break
  phi_old.assign(phi)

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
file_bis = File('verif_x.pvd')
phi_x = interpolate(phi.dx(0), W)
proj = project(inner(phi_x,phi_x), UU, name='test phi_x')
file_bis.write(proj)
file_ter = File('verif_y.pvd')
phi_y = interpolate(phi.dx(1), W)
proj = project(inner(phi_y,phi_y), UU, name='test phi_y')
file_ter.write(proj)
file_4 = File('verif_prod.pvd')
proj = project(inner(phi_x,phi_y), UU, name='test PS')
file_4.write(proj)

#Test
test = project(div(grad(phi)), W, name='minimal')
file_6 = File('minimal_bis.pvd')
file_6.write(test)
