"""
Douglas N. Arnold, 2014-12-03

Solve the Dirichlet problem for the minimal surface equation.

-div( q grad u) = 0,

where q = q(grad u) = (1 + |grad u|^2)^{-1/2}

using Newton's method as implemented in FEniCS with automatic differentiation
to compute the Jacobian.
"""

from fenics import *
import matplotlib.pyplot as plt

# the coefficient function
def q(u):
  return (1+inner(grad(u),grad(u)))**(-.5)

# Create mesh and define function space
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, 'Lagrange', 2)

# initial guess (its boundary values specify the Dirichlet boundary conditions)
# (larger coefficient in front of the sin term makes the problem "more nonlinear")
u0exp = Expression('a*sin(2.5*pi*x[1])*x[0]', a=.2, degree=5)
u0 = interpolate(u0exp, V)
plot(u0)
print('initial surface area:', assemble(sqrt(1+inner(grad(u0), grad(u0)))*dx))

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
