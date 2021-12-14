#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
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
size_ref = 5 #10 #degub: 5
nx,ny = int(size_ref*H/float(L)),size_ref
mesh = PeriodicRectangleMesh(nx, ny, L, H, direction='y', diagonal='crossed')
V = VectorFunctionSpace(mesh, "ARG", 5, dim=3)
#V = VectorFunctionSpace(mesh, "BELL", 5, dim=3) #faster

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
pen = 1e2
pen_term = pen/h**4 * inner(phi_t, psi) * (ds(1) + ds(2))
L = pen/h**4 * inner(phi_D, psi)  * (ds(1) + ds(2))
solve(laplace+pen_term == L, phi)

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
  #area = assemble(sqrt(1+inner(grad(u),grad(u)))*dx)
  print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  #print(assemble(0.5*(sign(sq_norm(phi.dx(0)) - 3)+1) * (sq_norm(phi.dx(0)) - 3) * dx))
  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  print('no convergence after {} Picard iterations'.format(iter+1))
else:
  print('convergence after {} Picard iterations'.format(iter+1))


#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
projected = project(phi, U, name='surface')

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

