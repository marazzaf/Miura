#coding: utf-8

from fenics import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ufl
import sys

L,l = 10,2*pi
size_ref = 100 #degub #100
Nx,Ny = size_ref,int(size_ref*L/l)
mesh = RectangleMesh(Point(-L/2,-l/2), Point(L/2, l/2), Nx, Ny, "crossed")
bnd = MeshFunction('size_t', mesh, 1)
bnd.set_all(0)

V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
phi = Function(V, name="surface")
psi = TestFunction(V)

#Dirichlet BC
theta = pi/2
z = Expression('2*sin(theta/2)*x[0]', theta=theta, degree=3)
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*x[0]*x[0] + 1)
phi_D = as_vector((rho*cos(x[1]), rho*sin(x[1]), z))

#creating the bc object
bc = DirichletBC(V, phi_D, bnd, 0)

#Writing energy. No constraint for now...
norm_phi_x = sqrt(inner(phi.dx(0), phi.dx(0)))
norm_phi_y = sqrt(inner(phi.dx(1), phi.dx(1)))

#bilinear form
a = (ufl.ln((1+0.5*norm_phi_x)/(1-0.5*norm_phi_x)) * (psi[0].dx(0)+psi[1].dx(0)+psi[2].dx(0)) - 4/norm_phi_y * (psi[0].dx(1)+psi[1].dx(1)+psi[2].dx(1))) * dx

#adding constraints with penalty
pen = 1e5
a += pen * ((1 - 0.25*norm_phi_x) * norm_phi_y - 1) * (-inner(phi.dx(0),psi.dx(0))*norm_phi_y**2 + inner(phi.dx(1),psi.dx(1))*(4-norm_phi_x**2)) * dx

#To add the inequality constraints
def ppos(x): #definition of positive part for inequality constraints
    return(x+abs(x))/2

#solving problem
#solve(a == 0, phi, bc, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

#Other solver
J = derivative(a, phi)
problem = NonlinearVariationalProblem(a, phi, bc, J)
solver  = NonlinearVariationalSolver(problem)

#Parameters
prm = solver.parameters
#info(prm, True) #to get info on parameters
prm["nonlinear_solver"] = "snes"

#Solving
solver.solve()


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

#Plotting the function
vec_phi_ref = phi.vector().get_local()
vec_phi = vec_phi_ref.reshape((3, len(vec_phi_ref) // 3))
vec_phi_aux = vec_phi_ref.reshape((len(vec_phi_ref) // 3, 3))

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in vec_phi_aux:
    ax.scatter(i[0], i[1], i[2], color='r')

#ax = fig.gca(projection='3d')
#ax.plot_surface(vec_phi[0,:], vec_phi[1,:], vec_phi[2,:], cmap=cm.coolwarm,linewidth=0, antialiased=False)

plt.savefig('plot_constraint.pdf')

