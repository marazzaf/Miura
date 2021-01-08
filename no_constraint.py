#coding: utf-8

from dolfin import *
import numpy as np

size_ref = 5 #debug
L,l = 1,1
mesh = RectangleMesh(Point(-L/2,-l/2), Point(L/2, l/2), size_ref, size_ref, "crossed")

V = FunctionSpace(mesh, 'CG', 1)

u = Function(V)
v = TestFunction(V)

#Dirichlet BC
R = 10 #radius of the outer circle
x = SpatialCoordinate(mesh)
theta = np.arctan2(x[0], x[1])
#phi = Expression(('R*cos(', 'R*'), R=R, degree = 2)
phi_D = as_vector((R*cos(theta), R*sin(theta)))

#Writing energy. No constraint for now...
