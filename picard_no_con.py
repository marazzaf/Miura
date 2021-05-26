from dolfin import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import sys
import ufl

# the coefficient functions
def p(phi):
  #return  1 / (1 - 0.25*inner(phi.dx(0), phi.dx(0)))
  return inner(phi.dx(1), phi.dx(1))**2 / 16 #test to integrate the equality constraint

def q(phi):
  #return 4 / inner(phi.dx(1), phi.dx(1))
  return 0.25 #test to integrate the equality constraint

# Create mesh and define function space
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta)))
alpha = sqrt(1 / (1 - sin(theta/2)**2))
l = 2*pi/alpha
size_ref = 5 #degub: 5
Nx,Ny = int(size_ref*l/float(L)/5),size_ref
mesh = RectangleMesh(Point(0,0), Point(L, l), Nx, Ny, "crossed")
# Define finite elements spaces and build mixed space
V = VectorElement("CG", mesh.ufl_cell(), 2, dim=3)
W = TensorElement("CG", mesh.ufl_cell(), 1, shape=(3,3))
Z = FunctionSpace(mesh, V * W)
V,W = Z.split()
#For projections
U = FunctionSpace(mesh, 'DG', 0)

# Reference solution
z = Expression('2*sin(theta/2)*x[0]', theta=theta, degree = 1)
rho = Expression('sqrt(4*pow(cos(theta/2),2)*x[0]*x[0] + 1)', theta=theta, degree = 1)
phi_D = Expression(('rho*cos(alpha*x[1])', 'rho*sin(alpha*x[1])', 'z'), alpha=alpha, rho=rho, theta=theta, z=z, degree = 5)
i_phi_D = interpolate(phi_D, V.collapse())

#initial guess (its boundary values specify the Dirichlet boundary conditions)
phi_old = interpolate(phi_D, V.collapse())

##checking stuff
#test_x = project(inner(phi_old.dx(0), phi_old.dx(0)), U)
#vec_x = test_x.vector().get_local()
#pb_min = np.where(vec_x < 0)[0]
#print(pb_min)
#print(len(pb_min))
#pb_max = np.where(vec_x > 3)[0]
#print(len(pb_max))
#print(len(pb_max)/len(vec_x)*100)
#sys.exit()
##print(vec_x[pb_min],vec_x[pb_max])
##print(min(vec_x), max(vec_x))
#test_y = project(inner(phi_old.dx(1), phi_old.dx(1)), U)
#vec_y = test_y.vector().get_local()
#pb_min = np.where(vec_y < 1)
#pb_max = np.where(vec_y > 4)
#print(vec_y[pb_min],vec_y[pb_max])
##print(min(vec_y), max(vec_y))
###assertions
##assert min(vec_x) > 0 and max(vec_x) < 3
##assert min(vec_y) > 1 and max(vec_y) < 4
test = project(inner(phi_old.dx(0), phi_old.dx(1)), U)
vec = test.vector().get_local()
print(min(abs(vec)), max(abs(vec)))

# Define variational problem for Picard iteration
n = FacetNormal(mesh)
n = as_vector((n[0], n[1], 0))
res = Function(Z)
psi,tau = TestFunctions(Z)
#Linear problem
phi_,sigma_ = TrialFunctions(Z)
a1 = (p(phi_old) * inner(sigma_[0,:].dx(0), psi) + q(phi_old) * inner(sigma_[1,:].dx(1), psi)) *dx
a2 = inner(phi_old,div(tau)) * dx
a3 = inner(sigma_,tau) * dx
a = a1 + a2 + a3
L = dot(dot(tau, n), i_phi_D) * ds
##Nonlinear problem
#a = (p(phi_old) * inner(psi.dx(0), phi.dx(0)) + q(phi_old) * inner(psi.dx(1), phi.dx(1))) * dx

# Picard iteration
tol = 1.0E-3
maxiter = 100
for iter in range(maxiter):
  #solve(a == 0, phi, DirichletBC(V, phi_D, DomainBoundary())) # compute next Picard iterate #solving non-linear problem
  solve(a == L, res, DirichletBC(V.collapse(), phi_D, DomainBoundary())) # compute next Picard iterate #solving linear problem

  eps = sqrt(abs(assemble(inner(grad(phi-phi_old),grad(phi-phi_old))*dx))) # check increment size as convergence test
  print('iteration{:3d}  H1 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  if eps < tol:
    break
  phi_old.assign(phi)

  ##checking stuff
  #test_x = project(inner(phi.dx(0), phi.dx(0)), U)
  #vec_x = test_x.vector().get_local()
  #print(min(vec_x), max(vec_x))
  #test_y = project(inner(phi.dx(1), phi.dx(1)), U)
  #vec_y = test_y.vector().get_local()
  #print(min(vec_y), max(vec_y))
  ##assertions
  #assert min(vec_x) > 0 and max(vec_x) < 3
  #assert min(vec_y) > 1 and max(vec_y) < 4
  test = project(inner(phi.dx(0), phi.dx(1)), U)
  vec = test.vector().get_local()
  print(min(abs(vec)), max(abs(vec)))

if eps > tol:
  print('no convergence after {} Picard iterations'.format(iter+1))
else:
  print('convergence after {} Picard iterations'.format(iter+1))

area = assemble(sqrt(1+inner(grad(phi),grad(phi)))*dx)
print('Computed area: %.5e' % area)
phi_old = interpolate(phi_D, V)
area = assemble(sqrt(1+inner(grad(phi_old),grad(phi_old)))*dx)
print('Ref area: %.5e' % area)
sys.exit()

#plotting solution
vec_phi = phi.vector().get_local()
vec_phi = vec_phi.reshape((len(vec_phi) // 3, 3))
vec_phi_D = interpolate(phi_D, V).vector().get_local()
vec_phi_D = vec_phi_D.reshape((len(vec_phi_D) // 3, 3))

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i,j in zip(vec_phi,vec_phi_D):
  ax.scatter(i[0], i[1], i[2], color='r')
  ax.scatter(j[0], j[1], j[2], color='b')
plt.show()
