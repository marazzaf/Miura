from dolfin import *
from ufl import as_tensor

kappa_s = 138.0       #conductivity [W/m K]
kappa_l = 80.0       #conductivity [W/m K]

mesh = UnitSquareMesh(10,10)

Space = FunctionSpace(mesh, 'P', 1)
T = project(Expression("x[0]+0.1*x[1]", degree=2), Space)

kappa_space = FunctionSpace(mesh, "DG", 0)
kappa = project(kappa_s,kappa_space)
kappa_s = as_tensor(kappa_s,())
kappa_l = as_tensor(kappa_l,())
Tmelting = 0.5
print(kappa.vector().get_local())
kappa.assign(project(conditional(gt(T, Tmelting), kappa_l, kappa_s), kappa_space))
print(kappa.vector().get_local())
