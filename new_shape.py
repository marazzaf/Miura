#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys
import numpy as np

# the coefficient functions
def p(phi):
  return  inner(phi.dx(0), phi.dx(0))**2
  #return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  return 4
  #return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

# Create mesh and define function space
theta = np.pi/2
##check the sizes of the mesh. Might be the problem with some constraints.
L = 2*np.sin(0.5*np.arccos(0.5/np.cos(0.5*theta))) #length of rectangle
alpha = np.sqrt(1 / (1 - np.sin(theta/2)**2))
H = 2*np.pi/alpha #height of rectangle
l = np.sin(theta/2)*L #total height of cylindre
modif = 0.1 #0.02 #variation at the top

#writing the matrix of the system
A = np.zeros((6,6))
#Filling-in by line
A[0,:] = np.array([L**2/4, 0, 0, -L/2, 0, 1])
A[1,:] = np.array([L**2/4, H**2/4, L*H/4, L/2, H/2, 1])
A[2,:] = np.array([L**2/4, 0, 0, L/2, 0, 1])
A[3,:] = np.array([0, H**2/4, 0, 0, H/2, 1])
A[4,:] = np.array([L**2/4, H**2, -L*H/2, -L/2, H, 1])
A[5,:] = np.array([L**2/4, H**2, L*H/2, L/2, H, 1])

#Corresponding right-hand side
b = np.array([-l, l, l*(1+modif), 0, -l, l*(1-modif)])

#solution
coeffs = np.linalg.solve(A,b)

def z(x,y):
    return coeffs[0]*x*x + coeffs[1]*y*y + coeffs[2]*x*y + coeffs[3]*x + coeffs[4]*y + coeffs[5]

#Loading mesh
num_computation = 1
mesh = Mesh('rectangle_%i.msh' % num_computation) #change mesh to not use the symmetry any longer
V = VectorFunctionSpace(mesh, "HER", 3, dim=3)

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)

# Boundary conditions
x = SpatialCoordinate(mesh)
z = z(x[0], x[1])
rho = sqrt(4*np.cos(theta/2)**2*x[0]*x[0] + 1)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

# Initial guess
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations
lin_rho = np.sqrt(4*np.cos(theta/2)**2*H*H + 1)
phi.project(as_vector((lin_rho*cos(alpha*x[1]), lin_rho*sin(alpha*x[1]), z))) #initial guess is a cylinder

##plotting initial guess
#vec = project(phi, U).vector().get_local()
#vec_phi_aux = vec.reshape((len(vec) // 3, 3))
##3d plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for i in vec_phi_aux:
#  ax.scatter(i[0], i[1], i[2], color='r')
#plt.show()
#sys.exit()

#Defining the bilinear forms
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e2
pen_term = pen/h**4 * inner(phi_t, psi) * ds #(ds(1) + ds(3))
a += pen_term
L = pen/h**4 * inner(phi_D, psi)  * ds #(ds(1) + ds(3))

#penalty for inequality constraint
#pen = 1e1
pen_ineq = pen * 0.5*(sign(1 - sq_norm(phi.dx(0)))+1) * inner(phi_t.dx(1), psi.dx(1)) * dx
a += pen_ineq

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
file = File('new_%i.pvd' % num_computation)
file.write(projected)

#check ineq constraints
W = FunctionSpace(mesh, 'CG', 2)
ineq_1 = 0.5*(1+sign(sq_norm(phi.dx(0)) - 3))
ineq_2 = 0.5*(1+sign(sq_norm(phi.dx(1)) - 4))
ineq_3 = 0.5*(1+sign(1 - sq_norm(phi.dx(1))))
constraint = ineq_1 + ineq_2 + ineq_3
constraint = project(constraint, W, name='constraint')
file = File('new_constraint_%i.pvd' % num_computation)
file.write(constraint)


#check eq constraint
C = CellVolume(mesh)
value = inner(phi.dx(0), phi.dx(1)) / C * dx
print(assemble(value))
#sys.exit()


#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

#Nice 3d plot
x = vec_phi_aux[:,0]
y = vec_phi_aux[:,1]
z = vec_phi_aux[:,2]
ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
plt.show()
sys.exit()

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in vec_phi_aux:
  ax.scatter(i[0], i[1], i[2], color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
plt.title('Miura ori')
plt.savefig('new_shape_%i.pdf' % num_computation)


