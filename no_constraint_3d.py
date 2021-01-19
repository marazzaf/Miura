#coding: utf-8

from fenics import *
#from fenics_adjoint import *
#from pyadjoint import Block
#from atanh_overloaded import atanh
import ufl
import sys

L,l = 10,2*pi
Nx,Ny = 5,10
mesh = RectangleMesh(Point(-L/2,-l/2), Point(L/2, l/2), Nx, Ny, "crossed")
bnd = MeshFunction('size_t', mesh, 1)
bnd.set_all(0)

V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
phi = Function(V, name="surface")
psi = TestFunction(V)

#Dirichlet BC
#x = SpatialCoordinate(mesh)
K = 1.9 #gaussian curvature of surface ?
theta0 = pi/3
#h_x = L // Nx
#h_y = l // Ny
#def R(i)
#    return sqrt(((4-K*K)**2 * ((i-(Nx+1)/2)*h_x)**2+4)/(4-K**2))
#end
#z[i,j] = K*(i-(Nx+1)/2)*dx
#x[i,j] = (R(i)+0.01*(i-1)*(Nx+1-i)*(j-1)*(Ny+1-j)*dx*dx*dy*dy)*cos(j*dy)
#y[i,j] = (R(i)+0.01*(i-1)*(Nx+1-i)*(j-1)*(Ny+1-j)*dx*dx*dy*dy)*sin(j*dy)
#phi_D = as_vector((x,y,z))
phi_D = Constant((0,0,0))


#creating the bc object
bc = DirichletBC(V, phi_D, bnd, 0)

#Writing energy. No constraint for now...
norm_phi_x = sqrt(inner(phi.dx(0), phi.dx(0)))
norm_phi_y = sqrt(inner(phi.dx(1), phi.dx(1)))


#bilinear form
a = (ufl.ln((1+0.5*norm_phi_x)/(1-0.5*norm_phi_x)) * (psi[0].dx(0)+psi[1].dx(0)+psi[2].dx(0)) - 4/norm_phi_y * (psi[0].dx(1)+psi[1].dx(1)+psi[2].dx(1))) * dx
#pen = 10
#a += pen * inner(phi.dx(0), phi.dx(1)) * (psi[0]+psi[1]) * dx + (1 - 0.25*norm_phi_x) * norm_phi_y * (psi[0]+psi[1]) * dx
#solving problem
solve(a == 0, phi, bc, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

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
interval_y = project(inner(phi.dx(1), phi.dx(1)), U)

# Save solution in VTK format
file = File("test/no_constraint.pvd")
file << phi
file << interval_x
file << interval_y

