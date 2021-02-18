#coding: utf-8

from dolfin import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import sys

#To determine domain
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta)))
l = 2*pi
size_ref = 5 #degub: 5
Nx,Ny = int(size_ref*l/float(L)),size_ref
mesh = RectangleMesh(Point(-L/2,0), Point(L/2, l), Nx, Ny, "crossed")
bnd = MeshFunction('size_t', mesh, 1)
bnd.set_all(0)
ds = ds(subdomain_data=bnd)

#Approximation Space
V = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
phi = Function(V, name="surface")
psi = TestFunction(V)

#Defining the boundaries
def top_down(x, on_boundary):
    tol = 1e-2
    return (near(x[1], 0, tol) and on_boundary) or (near(x[1], l, tol) and on_boundary)

def left(x, on_boundary):
    tol = 1e-2
    return near(x[0], -L/2, tol) and on_boundary

def right(x, on_boundary):
    tol = 1e-2
    return near(x[0], L/2, tol) and on_boundary #and x[1] > l/2

mirror_boundary = AutoSubDomain(top_down)
mirror_boundary.mark(bnd, 1)
left_boundary = AutoSubDomain(left)
left_boundary.mark(bnd, 2)
right_boundary = AutoSubDomain(right)
right_boundary.mark(bnd, 3)

#Dirichlet BC
x = SpatialCoordinate(mesh)
z = 2*sin(theta/2)*x[0]
alpha = sqrt(1 / (1 - sin(theta/2)**2))
rho = sqrt(4*cos(theta/2)**2*x[0]*x[0] + 1)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))
#phi = project(phi_D, V) #test
#phi_D = as_vector((rho, 0, z))

#creating the bc object
#bcs = DirichletBC(V, phi_D, bnd, 0) #only Dirichlet on Mirror BC
phi = project(phi_D, V) #initial guess is the solution
bc1 = DirichletBC(V, phi_D, top_down)
bc2 = DirichletBC(V, phi_D, left)
bc3 = DirichletBC(V, phi_D, right)
bcs = [bc1,bc2,bc3]
for bc in bcs:
    bc.apply(phi.vector()) #just applying it to get a better initial guess?

#Writing energy. No constraint for now...
norm_phi_x = sqrt(inner(phi.dx(0), phi.dx(0)))
norm_phi_y = sqrt(inner(phi.dx(1), phi.dx(1)))

#bilinear form
a1 = (phi[0].dx(0).dx(0)*psi[0] + phi[1].dx(0).dx(0)*psi[1] + phi[2].dx(0).dx(0)*psi[2]) / (1 - 0.25*norm_phi_x**2) * dx
a2 = 4/norm_phi_y**2 * (phi[0].dx(1).dx(1)*psi[0] + phi[1].dx(1).dx(1)*psi[1] + phi[2].dx(1).dx(1)*psi[2]) * dx
a = a1 + a2

##old one
#a1 = ln(abs((1+0.5*norm_phi_x)/(1-0.5*norm_phi_x))) * (psi[0].dx(0)+psi[1].dx(0)+psi[2].dx(0)) * dx #correct
#a2 = -4/norm_phi_y * (psi[0].dx(1)+psi[1].dx(1)+psi[2].dx(1)) * dx
#a = a1+a2

#adding equality constraint with penalty
pen = 1e5
G = (1 - 0.25*norm_phi_x**2)*norm_phi_y**2 - 1
c = G * ((4 - norm_phi_x**2) * (phi[0].dx(1)*psi[0].dx(1) + phi[1].dx(1)*psi[1].dx(1) + phi[2].dx(1)*psi[2].dx(1)) - norm_phi_y**2 * (phi[0].dx(0)*psi[0].dx(0) + phi[1].dx(0)*psi[1].dx(0) + phi[2].dx(0)*psi[2].dx(0))) * dx
#how to write least-squares version of c?

#Constraint on orthogonality
d = inner(phi.dx(0), phi.dx(1))**2 * dx
e = derivative(d, phi, psi)

#Adding Neumann BC
neumann_x = phi_D.dx(0)
test = grad(phi_D)
#print(test[0,0])
#print(test[0,1])
#print(test[2,0])
#sys.exit()
#neumann_x = as_vector((Dx(phi_D[0],0),Dx(phi_D[1],0),Dx(phi_D[2],0)))
neumann_y = phi_D.dx(1)
#neumann_x = test[:,0]
#print(neumann_x)
#neumann_y = test[:,1]
rhs_n = (dot(neumann_x, psi) + dot(neumann_y, psi)) * (ds(2) + ds(3))
#rhs_n = (phi_D[0].dx(0)*psi[0] + phi_D[1].dx(0)*psi[1] + phi_D[0].dx(1)*psi[0] + phi_D[1].dx(1)*psi[1] + z.dx(0)*psi[2] + z.dx(1)*psi[2]) * (ds(2)+ds(3))

tot = a
#tot = a + e + c# - rhs_n 

# Compute solution
#solve(tot == 0, phi, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

##Other solver
##dphi = TrialFunction(V)
#J = derivative(tot, phi) #, dphi)
#problem = NonlinearVariationalProblem(tot, phi, bcs, J)
#solver  = NonlinearVariationalSolver(problem)
#
##Parameters
#prm = solver.parameters
##info(prm, True) #to get info on parameters
#prm["nonlinear_solver"] = "newton" #"snes" #"newton"
#prm["newton_solver"]['relative_tolerance'] = 1e-6
#prm['newton_solver']['maximum_iterations'] = 100
#
##Solving
#solver.solve() 

##solution is okay?
##test = action(a, Constant((1,1,1)))
#test = assemble(a)
#test = assemble( as_vector((phi[0].dx(0).dx(0), phi[1].dx(0).dx(0), phi[2].dx(0).dx(0))) / (1 - 0.25*norm_phi_x**2) * dx + 4/norm_phi_y**2 * as_vector((phi[0].dx(1).dx(1), phi[1].dx(1).dx(1), phi[2].dx(1).dx(1))) * dx )
#print(test)
##print(test)
#sys.exit()

#Old
a1 = ln(abs((1+0.5*norm_phi_x)/(1-0.5*norm_phi_x))) * (psi[0].dx(0)+psi[1].dx(0)+psi[2].dx(0)) * dx #correct
a2 = -4/norm_phi_y * (psi[0].dx(1)+psi[1].dx(1)+psi[2].dx(1)) * dx
a = a1+a2
print(assemble(a)[:].sum())
sys.exit()

#solution verifies constraints?
ps = inner(phi.dx(0), phi.dx(1)) * dx
ps = assemble(ps)
print(ps) #should be 0
vol = CellVolume(mesh)
cons = (1 - 0.25*norm_phi_x**2) * norm_phi_y**2 * dx
cons = assemble(cons) / l / L
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
sys.exit()

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
plt.show()

