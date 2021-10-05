#coding: utf-8

from firedrake import *

N = 4
#mesh = UnitSquareMesh(N,N)
mesh = RectangleMesh(N,N,1,1)

U = VectorFunctionSpace(mesh, "HER", 3, dim=3)

du,v = TrialFunction(U),TestFunction(U)

a = inner(du, v) * dx

A = assemble(a).M.handle
print(A.size)