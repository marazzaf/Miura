#coding: utf-8

from dolfin import *

m = Mesh()
with XDMFFile("convergence_1.xdmf") as infile:
    infile.read(m)

print(m.hmax())
