#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys
import numpy as n

#MPI
rank = COMM_WORLD.rank

# the coefficient functions
def p(phi):
  #return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))**2
  return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  #return 4
  return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

#plot function
def Plot(f):
  fig, axes = plt.subplots()
  #levels = np.linspace(0, 1, 51)
  contours = tricontourf(f, axes=axes, cmap="inferno") #levels=levels
  axes.set_aspect("equal")
  fig.colorbar(contours)
  plt.show()
  return

# Create mesh and define function space
alpha = 1
L = 1/alpha * 0.9 #length of rectangle
H = 2*pi/alpha #height of rectangle
size_ref = 60 #20 #10 #degub: 5
nx,ny = int(size_ref*L/H),int(size_ref*H/L)
#mesh = PeriodicRectangleMesh(nx, ny, L, H, direction='y', diagonal='crossed')
mesh = RectangleMesh(nx, ny, L, H, diagonal='crossed')
V = VectorFunctionSpace(mesh, "ARG", 5, dim=3)
#V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 1)

# Boundary conditions
x = SpatialCoordinate(mesh)
rho = x[0]
z = x[0]
phi_D = as_vector((rho*cos(x[1]*alpha), rho*sin(x[1]*alpha), z))

# Creating function to store solution
phi = Function(V, name='solution')
phi_old = Function(V) #for iterations

#Defining the bilinear forms
#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e2
#lhs
pen_term = pen/h**2 * inner(phi_t, psi) * ds #h**4
a += pen_term
#rhs
L = pen/h**2 * inner(phi_D, psi) * ds #h**4

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
solve(laplace+pen_term == L, phi)

#testing bounded slope condition of initial guess
test = project(sq_norm(phi.dx(0)), UU)
#print(test.vector().array())
if rank == 0:
  print(max(test.vector()))
#sys.exit()
assert max(test.vector()) < 4 #not elliptic otherwise.

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
for iter in range(maxiter):
  #linear solve
  solve(a == L, phi) # compute next Picard iterate

  #ellipticity test
  test = project(sq_norm(phi.dx(0)), UU)
  value = max(test.vector())
  try:
    assert value < 4 #not elliptic otherwise.
  except AssertionError:
    if rank == 0:
      print('Bounded slope condition: %.2e' % value)
      Plot(test)
    #sys.exit()
  
  #convergence test 
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  print('no convergence after {} Picard iterations'.format(iter+1))
else:
  print('convergence after {} Picard iterations'.format(iter+1))


#For plot
projected = project(phi, U, name='surface')

#Write 2d results
file = File('new_%i.pvd' % size_ref)
file.write(projected)

#check eq constraint
res = interpolate((1 - 0.25 * inner(phi.dx(0), phi.dx(0))) * inner(phi.dx(1), phi.dx(1)) - 1, UU)
res = res.vector()
print(max(abs(max(res)), abs(min(res)))) #l-infinity
test = interpolate(Constant(1), UU)
res = errornorm((1 - 0.25 * inner(phi.dx(0), phi.dx(0))) * inner(phi.dx(1), phi.dx(1)), test, 'l2')
print(res) #l2


#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

#writing a file with points
points = open('points_%i.txt' % size_ref, 'w')
for i in vec_phi_aux:
  points.write('%.5e %.5e %.5e\n' % (i[0], i[1], i[2]))
points.close()
sys.exit()

##Nice 3d plot
#x = vec_phi_aux[:,0]
#y = vec_phi_aux[:,1]
#z = vec_phi_aux[:,2]
#ax = plt.figure().add_subplot(projection='3d')
#ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
#plt.show()
#sys.exit()


##3d plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for i in vec_phi_aux:
#  ax.scatter(i[0], i[1], i[2], color='r')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.show()
#plt.title('Miura ori')
#plt.savefig('new_shape_%i.pdf' % size_ref)
