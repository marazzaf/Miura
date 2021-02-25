from dolfin import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# the coefficient functions
def p(phi):
  return  1 / (1 - 0.25*inner(phi.dx(0), phi.dx(0)))

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

# Create mesh and define function space
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta)))
l = 2*pi
size_ref = 25 #degub: 5
Nx,Ny = int(size_ref*l/float(L)),size_ref
mesh = RectangleMesh(Point(-L/2,0), Point(L/2, l), Nx, Ny, "crossed")
V = VectorFunctionSpace(mesh, 'Lagrange', 1, dim=3)
U = FunctionSpace(mesh, 'Lagrange', 1)

# initial guess (its boundary values specify the Dirichlet boundary conditions)
z = Expression('2*sin(theta/2)*x[0]', theta=theta, degree = 5)
alpha = sqrt(1 / (1 - sin(theta/2)**2))
rho = Expression('sqrt(4*pow(cos(theta/2),2)*x[0]*x[0] + 1)', theta=theta, degree = 5)
phi_D = Expression(('rho*cos(alpha*x[1])', 'rho*sin(alpha*x[1])', 'z'), alpha=alpha, rho=rho, theta=theta, z=z, degree = 5)

phi_old = interpolate(phi_D, V)

# Define nonlinear weak formulation
tol = 1.e-6
u = u0  # most recently computed solution
v = TestFunction(V)
F = q(u) * inner(grad(u), grad(v)) * dx
# solve nonlinear problem using Newton's method.  Note that there
# are numerous parameters that can be used to control the Newton iteration
solve(F == 0, u, DirichletBC(V, u0exp, DomainBoundary()), \
              solver_parameters={"newton_solver": {"absolute_tolerance": tol}})

print('surface area:', assemble(sqrt(1+inner(grad(u),grad(u)))*dx))
plot(u) #, interactive=True)
plt.show()
