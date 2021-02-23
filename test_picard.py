from dolfin import *
import sys

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
V = VectorFunctionSpace(mesh, 'Lagrange', 2, dim=3)
U = FunctionSpace(mesh, 'Lagrange', 1)

# initial guess (its boundary values specify the Dirichlet boundary conditions)
# (larger coefficient in front of the sin term makes the problem "more nonlinear")
z = Expression('2*sin(theta/2)*x[0]', theta=theta, degree = 5)
alpha = sqrt(1 / (1 - sin(theta/2)**2))
rho = Expression('sqrt(4*pow(cos(theta/2),2)*x[0]*x[0] + 1)', theta=theta, degree = 5)
phi_D = Expression(('rho*cos(alpha*x[1])', 'rho*sin(alpha*x[1])', 'z'), alpha=alpha, rho=rho, theta=theta, z=z, degree = 5)

phi_old = interpolate(phi_D, V)
#plot(phi_old)
#print('initial surface area:', assemble(sqrt(1+inner(grad(phi_old),grad(phi_old)))*dx))

#test on phi_old
test_x = project(inner(phi_old.dx(0), phi_old.dx(0)), U)
vec_x = test_x.vector().get_local()
#print(min(vec_x), max(vec_x))
assert min(vec_x) > 0 and max(vec_x) < 3
test_y = project(inner(phi_old.dx(1), phi_old.dx(1)), U)
vec_y = test_y.vector().get_local()
assert min(vec_y) > 1 and max(vec_y) < 4

# Define variational problem for Picard iteration
phi = TrialFunction(V)
psi = TestFunction(V)
a = (p(phi_old) * inner(phi.dx(0).dx(0), psi) + q(phi_old) * inner(phi.dx(1).dx(1), psi)) * dx
L = Constant(0.)*psi[0]*dx
phi = Function(V)

#Coercivity test
f1 = Constant((1,2,10))
f2 = Constant((1,0,1))
t1 = interpolate(f1, V)
t2 = interpolate(f2, V)
value = assemble((p(phi_old) * inner(t1.dx(0).dx(0), t2) + q(phi_old) * inner(t1.dx(1).dx(1), t2)) * dx)
n1 = sqrt(assemble(inner(t1, t1) * dx))
n2 = sqrt(assemble(inner(t2, t2) * dx))
print(value / n1 / n2)
sys.exit()

# Picard iteration
tol = 1.0E-3
maxiter = 50
for iter in range(maxiter):
    solve(a == L, phi, DirichletBC(V, phi_D, DomainBoundary())) # compute next Picard iterate

    #checking stuff
    test_x = project(inner(phi.dx(0), phi.dx(0)), U)
    vec_x = test_x.vector().get_local()
    print(min(vec_x), max(vec_x))
    test_y = project(inner(phi.dx(1), phi.dx(1)), U)
    vec_y = test_y.vector().get_local()
    print(min(vec_y), max(vec_y))
    #assertions
    assert min(vec_x) > 0 and max(vec_x) < 3
    assert min(vec_y) > 1 and max(vec_y) < 4
    
    eps = sqrt(abs(assemble(inner(grad(phi-phi_old),grad(phi-phi_old))*dx))) # check increment size as convergence test
    #area = assemble(sqrt(1+inner(grad(u),grad(u)))*dx)
    print('iteration{:3d}  H1 seminorm of delta: {:10.2e}'.format(iter+1, eps))
    if eps < tol:
      break
    phi_old.assign(phi)

if eps > tol:
  print('no convergence after {} Picard iterations'.format(iter+1))
else:
  print('convergence after {} Picard iterations'.format(iter+1))

plot(phi, interactive=True)
