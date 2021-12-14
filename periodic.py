#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys
import numpy as np

# the coefficient functions
def p(phi):
  #return  inner(phi.dx(0), phi.dx(0))**2
  return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  #return 4
  return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

# Create mesh and define function space
theta = pi/2
##check the sizes of the mesh. Might be the problem with some constraints.
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle
l = sin(theta/2)*L #total height of cylindre
modif = 0.1 #0.02 #0.1 #0.02 #variation at the top

#Loading mesh
size_ref = 10 #20 #10 #degub: 5
nx,ny = int(size_ref*H/float(L)),size_ref
mesh = PeriodicRectangleMesh(nx, ny, L, H, direction='y', diagonal='crossed')
#V = VectorFunctionSpace(mesh, "ARG", 5, dim=3)
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)

# Boundary conditions
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)

# Initial guess
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations
lin_rho = sqrt(4*cos(theta/2)**2*L*L/4 + 1)
phi.project(as_vector((lin_rho*cos(alpha*x[1]), lin_rho*sin(alpha*x[1]), z))) #initial guess is a normal cylinder

#Defining the bilinear forms
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))
modif = 0.2
z = -l*modif*(x[1]-H/2)**2*4/H/H + l*(1+modif)
mod_phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))
h = CellDiameter(mesh)
pen = 1e3
#lhs
pen_term = pen/h**4 * inner(phi_t, psi) * (ds(1) + ds(2)) #Dirichlet BC top and bottom surfaces
a += pen_term
#rhs
L = pen/h**4 * inner(phi_D, psi) * ds(1) + pen/h**4 * inner(mod_phi_D, psi) * ds(2)

#Dirichlet on grad
n = FacetNormal(mesh)
pen_term = pen/h/h * inner(dot(grad(phi_t),n), dot(grad(psi),n)) * (ds(1) + ds(2)) #ds
a += pen_term
#grad(phi_D) has shape (3,2)
#grad_mod_phi_D = as_tensor(((rho.dx(0)*cos(alpha*x[1]), -alpha*rho*sin(alpha*x[1])), (rho.dx(0)*sin(alpha*x[1]), alpha*rho*cos(alpha*x[1])), (2*sin(theta/2), -2*l*modif*(x[1]-H/2)*4/H/H)))
L += pen/h/h * inner(dot(grad(phi_D),n), dot(grad(psi),n)) * (ds(1)+ds(2)) # + pen/h/h * inner(dot(grad_mod_phi_D,n), dot(grad(psi),n)) * ds(2) #ds

#penalty for inequality constraint
#pen = 1e1
pen_ineq = pen/h**4 * 0.5*(sign(1 - sq_norm(phi.dx(1)))+1) * inner(phi_t.dx(1), psi.dx(1)) * dx
#a += pen_ineq
pen_ineq = pen * 0.5*(sign(sq_norm(phi.dx(0)) - 3)+1) * inner(phi_t.dx(0), psi.dx(0)) * dx
#a += pen_ineq
pen_ineq = pen * 0.5*(sign(sq_norm(phi.dx(1)) - 4)+1) * inner(phi_t.dx(1), psi.dx(1)) * dx
#a += pen_ineq

#penalty for scalar product
pen = 1e1
pen_eq = pen * sign(inner(phi.dx(0), phi.dx(1))) * (inner(phi_t.dx(0), psi.dx(1)) + inner(phi_t.dx(0), psi.dx(1))) * dx
#a += pen_eq

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
for iter in range(maxiter):
  #linear solve
  solve(a == L, phi) # compute next Picard iterate
    
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  print('no convergence after {} Picard iterations'.format(iter+1))
else:
  print('convergence after {} Picard iterations'.format(iter+1))


#For plot
projected = project(phi, U, name='surface')

#Write 2d results
file = File('new_%i.pvd' % size_ref)
file.write(projected)

##check ineq constraints
#W = FunctionSpace(mesh, 'CG', 2)
#ineq_1 = 0.5*(1+sign(sq_norm(phi.dx(0)) - 3))
#ineq_2 = 0.5*(1+sign(sq_norm(phi.dx(1)) - 4))
#ineq_3 = 0.5*(1+sign(1 - sq_norm(phi.dx(1))))
#constraint = ineq_1 + ineq_2 + ineq_3
#constraint = project(constraint, W, name='constraint')
#file = File('new_constraint_%i.pvd' % size_ref)
#file.write(constraint)
#
#
##check eq constraint
#C = CellVolume(mesh)
#value = inner(phi.dx(0), phi.dx(1)) / C * dx
#print(assemble(value))
#sys.exit()


#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

#writing a file with points
points = open('points_%i.txt' % size_ref, 'w')
for i in vec_phi_aux:
  points.write('%.5e %.5e %.5e\n' % (i[0], i[1], i[2]))
points.close()
sys.exit()

##Nice 3d plot
#x = vec_phi_aux[:,0]
#y = vec_phi_aux[:,1]
#z = vec_phi_aux[:,2]
#ax = plt.figure().add_subplot(projection='3d')
#ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
#plt.show()
#sys.exit()


##3d plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for i in vec_phi_aux:
#  ax.scatter(i[0], i[1], i[2], color='r')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.show()
#plt.title('Miura ori')
#plt.savefig('new_shape_%i.pdf' % size_ref)
