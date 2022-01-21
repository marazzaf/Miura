#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys

# the coefficient functions
def p(phi):
  return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

# Size for the domain
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta))) #length of rectangle
alpha = sqrt(1 / (1 - sin(theta/2)**2))
H = 2*pi/alpha #height of rectangle
l = sin(theta/2)*L

#Creating mesh
size_ref = 50 #10 #degub: 5
nx,ny = int(size_ref*H/float(L)),size_ref
#mesh = PeriodicRectangleMesh(nx, ny, L, H, direction='y', diagonal='crossed')
mesh = RectangleMesh(nx, ny, L, H)
#mesh = Mesh('convergence_5.msh')
#V = VectorFunctionSpace(mesh, "ARG", 5, dim=3)
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3) #faster
VV = FunctionSpace(mesh, 'CG', 4)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#  Dirichlet boundary conditions
x = SpatialCoordinate(mesh)
rho = sqrt(4*cos(theta/2)**2*(x[0]-L/2)**2 + 1)
z = 2*sin(theta/2) * (x[0]-L/2)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

#initial guess
#solve laplace equation on the domain
phi = Function(V, name='solution')
phi_t = TrialFunction(V)
psi = TestFunction(V)
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
#penalty term for Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1 #1e2
pen_term = pen/h**4 * inner(phi_t, psi) * ds #(ds(1) + ds(2))
L = pen/h**4 * inner(phi_D, psi)  * ds #(ds(1) + ds(2))
#pen_term = pen * inner(phi_t, psi) * ds #(ds(1) + ds(2))
#L = pen * inner(phi_D, psi)  * ds #(ds(1) + ds(2))
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
#solve(A, phi, b, solver_parameters={'ksp_type': 'cg','pc_type': 'bjacobi', 'ksp_rtol': 1e-5})
PETSc.Sys.Print('Laplace equation ok')

#Writing our problem now
#bilinear form for linearization
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
#pen_term = pen/h**4 * inner(phi_t, psi) * ds
a += pen_term
#L = pen/h**4 * inner(phi_D, psi)  * ds

# Picard iteration
tol = 1e-5 #1e-9
maxiter = 50
phi_old = Function(V) #for iterations
for iter in range(maxiter):
  #linear solve
  A = assemble(a)
  b = assemble(L)
  pp = interpolate(p(phi), VV)
  PETSc.Sys.Print('Min of p: %.3e' % pp.vector().array().min())
  #solve(A, phi, b, solver_parameters={'ksp_type': 'cg','pc_type': 'bjacobi', 'ksp_rtol': 1e-5})
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate
    
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))

  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Computing error
X = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
projected = interpolate(div(grad(phi)), X)
ref = interpolate(div(grad(phi_D)), X)
#err = errornorm(projected, ref, 'l2')
err = sqrt(assemble(inner(div(grad(phi-phi_D)), div(grad(phi-phi_D)))*dx))
PETSc.Sys.Print('Error: %.3e' % err)

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
projected = project(phi, U, name='surface')
#
##Checking if second equation is verified
#C = CellVolume(mesh)
#phi_x = project(phi.dx(0), U)
#phi_y = project(phi.dx(1), U)
#res = ((1 - 0.25 * inner(phi_x, phi_x)) * inner(phi_y, phi_y) - 1) / C * dx
#print(abs(assemble(res))) #l1
#UU = FunctionSpace(mesh, 'CG', 1)
#test = interpolate(Constant(1), UU)
#res = errornorm((1 - 0.25 * inner(phi_x, phi_x)) * inner(phi_y, phi_y), test, 'l2')
#print(res) #l2
#res = interpolate((1 - 0.25 * inner(phi.dx(0), phi.dx(0))) * inner(phi.dx(1), phi.dx(1)) - 1, UU)
#res = res.vector()
#print(max(abs(max(res)), abs(min(res)))) #l-infinity
#sys.exit()

#Write 2d results
file = File('hyper.pvd')
file.write(projected)

#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

##Nice 3d plot
#x = vec_phi_aux[:,0]
#y = vec_phi_aux[:,1]
#z = vec_phi_aux[:,2]
#ax = plt.figure().add_subplot(projection='3d')
#ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
#plt.title('Miura ori')
#plt.savefig('miura.pdf')
#plt.show()
#sys.exit()

#reference
ref = project(phi_D, U, name='ref')
vec_ref = ref.vector().get_local()
vec_ref = vec.reshape((len(vec_ref) // 3, 3))
file_bis = File('ref_hyper.pvd')
file_bis.write(ref)

#magnitude diff
#img = plot(sqrt(dot(projected-ref, projected-ref)))
#plt.colorbar(img)
#plt.show()
#diff = Function(U, name='diff')
#diff.vector()[:] = projected.vector() - ref.vector()
#file_bis = File('diff.pvd')
#file_bis.write(diff)
#sys.exit()


##3d plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
##for i,j in zip(vec_phi_aux,vec_ref):
#for i in vec_phi_aux:
#  ax.scatter(i[0], i[1], i[2], color='r')
#  #ax.scatter(j[0], j[1], j[2], color='b')
#plt.show()

#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

#writing a file with points
points = open('hyperboloid_%i.txt' % size_ref, 'w')
for i in vec_phi_aux:
  points.write('%.5e %.5e %.5e\n' % (i[0], i[1], i[2]))
points.close()
sys.exit()

#computing normals and writing them
normals = open('normals_%i.txt' % size_ref, 'w')
phi_x = project(phi.dx(0), U).vector().get_local()
phi_x = phi_x.reshape((len(vec) // 3, 3))
phi_y = project(phi.dx(1), U).vector().get_local()
phi_y = phi_y.reshape((len(vec) // 3, 3))
import numpy as np
for i,j in zip(phi_x,phi_y):
  normal = -np.cross(i,j)
  normal /= np.linalg.norm(normal)
  normals.write('%.5e %.5e %.5e\n' % (normal[0], normal[1], normal[2]))
normals.close()
sys.exit()

