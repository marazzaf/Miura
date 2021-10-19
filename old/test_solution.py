#coding: utf-8

from fenics import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import ufl
import sys

#To determine domain
theta = pi/2
L = Constant(2*sin(0.5*acos(0.5/cos(0.5*theta))))
l = 2*pi
size_ref = 200 #degub: 5
Nx,Ny = int(size_ref*l/float(L)),size_ref
mesh = RectangleMesh(Point(-L/2,0), Point(L/2, l), Nx, Ny, "crossed")
bnd = MeshFunction('size_t', mesh, 1)
bnd.set_all(0)
ds = ds(subdomain_data=bnd)

#Approximation Space
V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
phi = Function(V, name="surface")
psi = TestFunction(V)

#Defining the boundaries
def top_down(x, on_boundary):
    tol = 1e-2
    return (near(x[1], 0, tol) and on_boundary) or (near(x[1], l, tol) and on_boundary)

def left(x, on_boundary):
    tol = 1e-2
    return (near(x[0], -L/2, tol) and on_boundary)

def part_right(x, on_boundary):
    tol = 1e-2
    return (near(x[0], L/2, tol) and x[1] > l/2 and on_boundary)

#Dirichlet BC
z = Expression('2*sin(theta/2)*x[0]', theta=theta, degree=3)
alpha = sqrt(1 / (1 - sin(theta/2)**2))
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*x[0]*x[0] + 1)
#rho = sqrt(4*cos(theta/2)**2*x[0]*x[0] - sin(theta/2)**2*x[0] + 1) #test
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))
phi = project(phi_D, V) #test

#checking intervals
U = FunctionSpace(mesh, 'CG', 1)
interval_x = project(inner(phi.dx(0), phi.dx(0)), U)
vec_interval_x = interval_x.vector().get_local()
#print(min(vec_interval_x),max(vec_interval_x))
interval_y = project(inner(phi.dx(1), phi.dx(1)), U)
vec_interval_y = interval_y.vector().get_local()
#print(min(vec_interval_y),max(vec_interval_y))

phi = project(phi_D,V)
norm_phi_x = sqrt(inner(phi.dx(0), phi.dx(0)))
norm_phi_y = sqrt(inner(phi.dx(1), phi.dx(1)))

#solution verifies constraints?
ps = inner(phi.dx(0), phi.dx(1)) * dx
ps = assemble(ps)
#print(ps) #should be 0
vol = CellVolume(mesh)
cons = (1 - 0.25*norm_phi_x) * norm_phi_y * dx
cons = assemble(cons)
#print(cons / float(l) / float(L)) #should be one
#sys.exit()

#bilinear form
a1 = ufl.ln((1+0.5*norm_phi_x)/(1-0.5*norm_phi_x)) * (psi[0].dx(0)+psi[1].dx(0)+psi[2].dx(0)) * dx
a2 = -4/norm_phi_y * (psi[0].dx(1)+psi[1].dx(1)+psi[2].dx(1)) * dx
a = a1 + a2

#Tests
phi = project(phi_D,V)
print(assemble(a1).get_local())
print(assemble(a).get_local())

#adding equality constraint with penalty
b = ((1 - 0.25*inner(phi.dx(0), phi.dx(0))) * inner(phi.dx(1), phi.dx(1)) - 1)**2 * dx #least-squares penalty on equality constraint
c = derivative(b, phi, psi)

print(assemble(b))
sys.exit()

#To add the inequality constraints
def ppos(x): #definition of positive part for inequality constraints
    return(x+abs(x))/2

#adding inequality constraint with penalty
#d = pen * (ppos(norm_phi_x**2 - 3) + ppos(norm_phi_y**2 - 4) + ppos(1 - norm_phi_y**2)) * dx
d = pen * (ppos(norm_phi_x - sqrt(3))**2 + ppos(norm_phi_y - 2)**2 + ppos(1 - norm_phi_y)**2) * dx
e = derivative(d, phi)

print(assemble(d).get_local())