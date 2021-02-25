from dolfin import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# the coefficient functions
def p(phi):
  return  1 / (1 - 0.25*(phi[0].dx(0) * phi[0].dx(0) + phi[1].dx(0) * phi[1].dx(0) + phi[2].dx(0) * phi[2].dx(0)))

def q(phi):
  return 4 / (phi[0].dx(1) * phi[0].dx(1) + phi[1].dx(1) * phi[1].dx(1) + phi[2].dx(1) * phi[2].dx(1))

# Create mesh and define function space
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta)))
l = 2*pi
size_ref = 20 #degub: 5
Nx,Ny = int(size_ref*l/float(L)),size_ref
mesh = RectangleMesh(Point(-L/2,0), Point(L/2, l), Nx, Ny, "crossed")
V = VectorFunctionSpace(mesh, 'Lagrange', 2, dim=6)

#Defining the boundaries
def top_down(x, on_boundary):
    tol = 1e-2
    return (near(x[1], 0, tol) and on_boundary) or (near(x[1], l, tol) and on_boundary)

def left(x, on_boundary):
    tol = 1e-2
    return near(x[0], -L/2, tol) and on_boundary

def right(x, on_boundary):
    tol = 1e-2
    return near(x[0], L/2, tol) and on_boundary

# Dirichlet boundary conditions
z = Expression('2*sin(theta/2)*x[0]', theta=theta, degree = 5)
alpha = sqrt(1 / (1 - sin(theta/2)**2))
rho = Expression('sqrt(4*pow(cos(theta/2),2)*x[0]*x[0] + 1)', theta=theta, degree = 5)
phi_D = Expression(('rho*cos(alpha*x[1])', 'rho*sin(alpha*x[1])', 'z', '0', '0', '0'), alpha=alpha, rho=rho, z=z, degree = 5)
phi_D_x = Expression('rho*cos(alpha*x[1])', alpha=alpha, rho=rho, degree = 5)
phi_D_y = Expression('rho*sin(alpha*x[1])', alpha=alpha, rho=rho, degree = 5)

#Dirichlet BC
bc1 = DirichletBC(V.sub(0), phi_D_x, DomainBoundary())
bc2 = DirichletBC(V.sub(1), phi_D_y, DomainBoundary())
bc3 = DirichletBC(V.sub(2), z, DomainBoundary())
bcs = [bc1,bc2,bc3]
#bcs = bc1

#initial guess
phi_old = interpolate(phi_D, V)

# Define nonlinear weak formulation
tol = 1e-5
atol = 1e-6
#phi = phi_old
phi = TrialFunction(V)
psi = TestFunction(V)
F = (p(phi_old) * (psi[0].dx(0) * phi[0].dx(0) + psi[1].dx(0) * phi[1].dx(0) + psi[2].dx(0) * phi[2].dx(0)) + q(phi_old) * (psi[0].dx(1) * phi[0].dx(1) + psi[1].dx(1) * phi[1].dx(1) + psi[2].dx(1) * phi[2].dx(1))) * dx

#Adding pen
def ppos(x): #definition of positive part for inequality constraints
    return(x+abs(x))/2
norm_phi_x = (phi[0].dx(0) * phi[0].dx(0) + phi[1].dx(0) * phi[1].dx(0) + phi[2].dx(0) * phi[2].dx(0))
norm_phi_y = (phi[0].dx(1) * phi[0].dx(1) + phi[1].dx(1) * phi[1].dx(1) + phi[2].dx(1) * phi[2].dx(1))
G = psi[3] * ppos(norm_phi_x - sqrt(3)) * dx + psi[4] * ppos(norm_phi_y - 2) * dx + psi[5] * ppos(1 - norm_phi_y) * dx

#Adding pen
#F = F + G

# solve nonlinear problem using Newton's method.  Note that there
# are numerous parameters that can be used to control the Newton iteration
#solve(F == 0, phi, bcs, solver_parameters={"newton_solver": {"absolute_tolerance": atol, "relative_tolerance": tol}}) #nonlinear
L = Constant(0)*psi[0]*dx
phi = Function(V)
solve(F == L, phi, bcs) #linear
#sys.exit()

#plotting solution
vec_phi_ref = phi.vector().get_local()
vec_phi_aux = vec_phi_ref.reshape((len(vec_phi_ref) // 6, 6))

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in vec_phi_aux:
  ax.scatter(i[0], i[1], i[2], color='r')
plt.show()
