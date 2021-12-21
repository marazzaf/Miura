#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys

# the coefficient functions
def p(phi):
  return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))**2
  #return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  return 4
  #return 4 / inner(phi.dx(1), phi.dx(1))

# Size for the domain
H = 0.6 #length of rectangle
L = pi #height of rectangle

#Creating mesh
size_ref = 1 #coarse
mesh = Mesh('rectangle_mobius.msh')
V = VectorFunctionSpace(mesh, "ARG", 5, dim=3)
#V = VectorFunctionSpace(mesh, "BELL", 5, dim=3) #faster
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3) #for projection
UU = FunctionSpace(mesh, 'CG', 1) #projection for scalars

#  Dirichlet boundary conditions
x = SpatialCoordinate(mesh)
c = as_vector((cos(2*x[0]), sin(2*x[0]), 0))
r = as_vector((cos(2*x[0])*cos(2*x[0]), cos(2*x[0])*sin(2*x[0]), sin(x[0])))
phi_D = c + x[1] * r

#initial guess
#solve laplace equation on the domain
phi = Function(V, name='solution')
phi_t = TrialFunction(V)
psi = TestFunction(V)
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
#penalty term for Dirichlet BC
h = CellDiameter(mesh)
pen = 1e2
pen_term = pen/h**4 * inner(phi_t, psi) * (ds(1) + ds(2))
L = pen/h**4 * inner(phi_D, psi)  * (ds(1) + ds(2))
solve(laplace+pen_term == L, phi)

#Checking if verifying a bounded slope condition
print(max(interpolate(inner(phi.dx(0), phi.dx(0)), UU).vector()))
sys.exit()

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
  solve(a == L, phi) # compute next Picard iterate
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))

  #test to see if bounded slope ok
  print(max(interpolate(inner(phi.dx(0), phi.dx(0)), UU).vector()))

  #test to finish computation
  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  print('no convergence after {} Picard iterations'.format(iter+1))
else:
  print('convergence after {} Picard iterations'.format(iter+1))


#For projection
projected = project(phi, U, name='surface')

#Checking if second equation is verified
C = CellVolume(mesh)
phi_x = project(phi.dx(0), U)
phi_y = project(phi.dx(1), U)
res = ((1 - 0.25 * inner(phi_x, phi_x)) * inner(phi_y, phi_y) - 1) / C * dx
print(abs(assemble(res))) #l1
test = interpolate(Constant(1), UU)
res = errornorm((1 - 0.25 * inner(phi_x, phi_x)) * inner(phi_y, phi_y), test, 'l2')
print(res) #l2
res = interpolate((1 - 0.25 * inner(phi.dx(0), phi.dx(0))) * inner(phi.dx(1), phi.dx(1)) - 1, UU)
res = res.vector()
print(max(abs(max(res)), abs(min(res)))) #l-infinity

#Write 2d results
file = File('mobius.pvd')
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

