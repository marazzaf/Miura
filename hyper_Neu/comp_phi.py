#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
import sys

def comp_phi(mesh, Grad):
    W = VectorFunctionSpace(mesh, "CG", 3, dim=3)

    #Reconstructing phi
    phi = Function(W, name='solution')
    phi_t = TrialFunction(W)
    psi = TestFunction(W)
    aux = inner(grad(phi_t), grad(psi)) * dx #g_psi in another space?
    L = inner(Grad, grad(psi)) * dx

    #solving
    A = assemble(aux) #, bcs=bcs_test)
    b = assemble(L) #, bcs=bcs_test)
    solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})

    return phi

#Testing now
def test():
    # Size for the domain
    theta = pi/2 #pi/2
    L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
    alpha = sqrt(1 / (1 - sin(theta/2)**2))
    H = 2*pi/alpha #height of rectangle

    #Creating mesh
    size_ref = 20 #degub: 5
    mesh = PeriodicRectangleMesh(size_ref, size_ref, L, H, direction='y', diagonal='crossed')

    #  Ref solution
    x = SpatialCoordinate(mesh)
    rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
    z = 2*sin(theta/2) * (x[0]-L/2)
    phi_ref = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

    #Reconstruction
    phi = comp_phi(mesh, grad(phi_ref))
    #Write 3d results
    file = File('reconstruction_test.pvd')
    Z = VectorFunctionSpace(mesh, "CG", 3, dim=3)
    projected = Function(Z, name='surface')
    projected.interpolate(phi - as_vector((x[0], x[1], 0)))
    file.write(projected)

#test()
