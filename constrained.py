#coding: utf-8

from fenics import *
#from fenics_adjoint import *
#from pyadjoint import Block
#from atanh_overloaded import atanh
import ufl
import sys

size_ref = 50 #5 for debug
L,l = Constant(10),Constant(2*pi)
mesh = RectangleMesh(Point(-L/2,-l/2), Point(L/2, l/2), size_ref, size_ref, "crossed")
bnd = MeshFunction('size_t', mesh, 1)
bnd.set_all(0)

V = VectorFunctionSpace(mesh, 'CG', 1)
phi = Function(V, name="surface")
psi = TestFunction(V)

#Dirichlet BC
R = 10 #radius of the outer circle
r = 1
rho = Expression('r + (R-r)/R*x[0]', r=r, R=R, degree=1)
alpha = 1
#theta = ufl.atan_2(x[0], x[1])
phi_D = rho * Expression(('cos(alpha*x[1])', 'sin(alpha*x[1])'), alpha=alpha, degree=3)

#creating the bc object
bc = DirichletBC(V, phi_D, bnd, 0)

#Writing energy. No constraint for now...
norm_phi_x = sqrt(inner(phi.dx(0), phi.dx(0)))
norm_phi_y = sqrt(inner(phi.dx(1), phi.dx(1)))

#bilinear form
#a = (ufl.ln((1+0.5*norm_phi_x)/(1-0.5*norm_phi_x)) * (psi[0].dx(0)+psi[1].dx(0)) - 4/norm_phi_y * (psi[0].dx(1)+psi[1].dx(1))) * dx
#pen = 1e5
a = inner(phi.dx(0), phi.dx(1)) * (psi[0]+psi[1]) * dx + ((1 - 0.25*norm_phi_x) * norm_phi_y - 1) * (psi[0]+psi[1]) * dx
#solving problem
solve(a == 0, phi, bc, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

#Assemble Minimisation problem
J = assemble(0.5 * inner(y - yd, y - yd) * dx + nu / 2 * inner(u, u) * dx)

# Formulate the reduced problem
m = Control(u)  # Create a parameter from u, as it is the variable we want to optimise
Jhat = ReducedFunctional(J, m)

#Add the bounds for derivatives. Variational inequality?

#solution verifies constraints?
ps = inner(phi.dx(0), phi.dx(1)) * dx
ps = assemble(ps)
print(ps) #should be 0
cons = (1 - 0.25*norm_phi_x) * norm_phi_y * dx
cons = assemble(cons)
print(cons) #should be one

#checking intervals
U = FunctionSpace(mesh, 'CG', 1)
interval_x = project(inner(phi.dx(0), phi.dx(0)), U)
vec_interval_x = interval_x.vector().get_local()
print(min(vec_interval_x),max(vec_interval_x))
interval_y = project(inner(phi.dx(1), phi.dx(1)), U)
vec_interval_y = interval_y.vector().get_local()
print(min(vec_interval_y),max(vec_interval_y))

# Save solution in VTK format
file = File("test/no_constraint.pvd")
file << phi
file << interval_x
file << interval_y

