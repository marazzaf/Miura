#coding: utf-8

from dolfin import *
import sys

#Domain of study
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta)))
l = 2*pi
size_ref = 100 #degub: 5
Nx,Ny = int(size_ref*l/float(L)),size_ref
mesh = RectangleMesh(Point(-L/2,0), Point(L/2, l), Nx, Ny, "crossed")

#Function of study
x = SpatialCoordinate(mesh)
Q = (1 - 0.25*x[0]*x[0])*x[1]*x[1] - 1
P = Q*Q
grad_P = grad(P)
hess_P = grad(grad_P)
det_P = det(hess_P)

#Study of the sign of Q
U = FunctionSpace(mesh, 'CG', 1)
vec = project(Q, U).vector().get_local()
print(min(vec), max(vec))
grad_Q = grad(Q)
hess_Q = grad(grad_Q)
det_Q = det(hess_Q)
vec = project(det_Q, U).vector().get_local()
print(min(vec), max(vec))
sys.exit()

#Study of the sign of det
vec = project(det_P, U).vector().get_local()
print(min(vec), max(vec))
sys.exit()

#Reference solution
x = SpatialCoordinate(mesh)
z = 2*sin(theta/2)*x[0]
alpha = sqrt(1 / (1 - sin(theta/2)**2))
rho = sqrt(4*cos(theta/2)**2*x[0]*x[0] + 1)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))
