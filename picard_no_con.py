from dolfin import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys
import ufl

# the coefficient functions
def p(phi):
  return  1 / (1 - 0.25*inner(phi.dx(0), phi.dx(0)))

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

# Create mesh and define function space
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta)))
alpha = sqrt(1 / (1 - sin(theta/2)**2))
l = 2*pi/alpha
size_ref = 100 #degub: 5
Nx,Ny = int(size_ref*l/float(L)),size_ref
mesh = RectangleMesh(Point(0,0), Point(L, l), Nx, Ny, "crossed")
V = VectorFunctionSpace(mesh, 'Lagrange', 2, dim=3)
U = FunctionSpace(mesh, 'Lagrange', 2)

# Reference solution
z = Expression('2*sin(theta/2)*x[0]', theta=theta, degree = 5)
rho = Expression('sqrt(4*pow(cos(theta/2),2)*x[0]*x[0] + 1)', theta=theta, degree = 5)
phi_D = Expression(('rho*cos(alpha*x[1])', 'rho*sin(alpha*x[1])', 'z'), alpha=alpha, rho=rho, theta=theta, z=z, degree = 5)

#initial guess (its boundary values specify the Dirichlet boundary conditions)
phi_old = interpolate(phi_D, V)

#checking stuff
test_x = project(inner(phi_old.dx(0), phi_old.dx(0)), U)
vec_x = test_x.vector().get_local()
print(min(vec_x), max(vec_x))
test_y = project(inner(phi_old.dx(1), phi_old.dx(1)), U)
vec_y = test_y.vector().get_local()
print(min(vec_y), max(vec_y))
#assertions
assert min(vec_x) > 0 and max(vec_x) < 3
assert min(vec_y) > 1 and max(vec_y) < 4

# Define variational problem for Picard iteration
phi = Function(V)
psi = TestFunction(V)
#Linear problem
phi_ = TrialFunction(V)
a = (p(phi_old) * inner(psi.dx(0), phi_.dx(0)) + q(phi_old) * inner(psi.dx(1), phi_.dx(1))) * dx
L = Constant(0.)*psi[0]*dx
##Nonlinear problem
#a = (p(phi_old) * inner(psi.dx(0), phi.dx(0)) + q(phi_old) * inner(psi.dx(1), phi.dx(1))) * dx

# Picard iteration
tol = 1.0E-3
maxiter = 50
for iter in range(maxiter):
  #solve(a == 0, phi, DirichletBC(V, phi_D, DomainBoundary())) # compute next Picard iterate #solving non-linear problem
  solve(a == L, phi, DirichletBC(V, phi_D, DomainBoundary())) # compute next Picard iterate #solving linear problem

  eps = sqrt(abs(assemble(inner(grad(phi-phi_old),grad(phi-phi_old))*dx))) # check increment size as convergence test
  #area = assemble(sqrt(1+inner(grad(u),grad(u)))*dx)
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

if eps > tol:
  print('no convergence after {} Picard iterations'.format(iter+1))
else:
  print('convergence after {} Picard iterations'.format(iter+1))

plot(phi, interactive=True)

  ##plotting solution
  #vec_phi_ref = phi.vector().get_local()
  #vec_phi = vec_phi_ref.reshape((3, len(vec_phi_ref) // 3))
  #vec_phi_aux = vec_phi_ref.reshape((len(vec_phi_ref) // 3, 3))

  ##3d plot
  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')
  #for i in vec_phi_aux:
  #  ax.scatter(i[0], i[1], i[2], color='r')
  #plt.show()
  #sys.exit()
