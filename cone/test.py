#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from firedrake.petsc import PETSc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys

# the coefficient functions
def p(phi):
  #return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))**2
  return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  #return 4
  return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

##plot function
#def Plot(f):
#  fig, axes = plt.subplots()
#  contours = tricontourf(f, axes=axes, cmap="inferno") #levels=levels
#  axes.set_aspect("equal")
#  fig.colorbar(contours)
#  plt.show()
#  return

# Create mesh and define function space
alpha = 1
L = 1/alpha #length of rectangle
H = pi/alpha #height of rectangle
size_ref = 40 #60 #20 #10 #degub: 5
#nx,ny = int(size_ref*L/H),int(size_ref*H/L)
#mesh = PeriodicRectangleMesh(nx, ny, L, H, direction='y', diagonal='crossed')
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')
#V = VectorFunctionSpace(mesh, "ARG", 5, dim=3)
V = VectorFunctionSpace(mesh, "BELL", 5, dim=3)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
UU = FunctionSpace(mesh, 'CG', 4)

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
pen = 1e1 #1e2
#lhs
pen_term = pen/h**4 * inner(phi_t, psi) * ds #h**2
a += pen_term
#rhs
L = pen/h**4 * inner(phi_D, psi) * ds #h**2

#Computing initial guess
laplace = inner(grad(phi_t), grad(psi)) * dx #laplace in weak form
A = assemble(laplace+pen_term)
b = assemble(L)
solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'})
PETSc.Sys.Print('Laplace equation ok')

#testing bounded slope condition of initial guess
test = project(sq_norm(phi.dx(0)), UU)
with test.dat.vec_ro as v:
    value = v.max()[1]
#value = max(test.vector())
#sys.exit()
try:
  assert value < 4 #not elliptic otherwise.
except AssertionError:
  PETSc.Sys.Print('Bouned slope condition %.2e' % value)

# Picard iterations
tol = 1e-5 #1e-9
maxiter = 50
for iter in range(maxiter):
  #linear solve
  A = assemble(a)
  b = assemble(L)
  pp = interpolate(p(phi), UU)
  PETSc.Sys.Print('Min of p: %.3e' % pp.vector().array().min())
  solve(A, phi, b, solver_parameters={'direct_solver': 'mumps'}) # compute next Picard iterate

  #ellipticity test
  test = project(sq_norm(phi.dx(0)), UU)
  with test.dat.vec_ro as v:
    value = v.max()[1]
  try:
    assert value < 4 #not elliptic otherwise.
  except AssertionError:
    PETSc.Sys.Print('Bouned slope condition %.2e' % value)
    #Plot(test)
    #sys.exit()
  
  #convergence test 
  eps = sqrt(assemble(inner(div(grad(phi-phi_old)), div(grad(phi-phi_old)))*dx)) # check increment size as convergence test
  PETSc.Sys.Print('iteration{:3d}  H2 seminorm of delta: {:10.2e}'.format(iter+1, eps))
  if eps < tol:
    break
  phi_old.assign(phi)

if eps > tol:
  PETSc.Sys.Print('no convergence after {} Picard iterations'.format(iter+1))
else:
  PETSc.Sys.Print('convergence after {} Picard iterations'.format(iter+1))

#Write 2d results
file = File('new_%i.pvd' % size_ref)
x = SpatialCoordinate(mesh)
projected = project(phi - as_vector((x[0], x[1], 0)), U, name='surface')
file.write(projected)
sys.exit()

#For plot
projected = project(phi, U, name='surface')

#check eq constraint
res = interpolate((1 - 0.25 * inner(phi.dx(0), phi.dx(0))) * inner(phi.dx(1), phi.dx(1)) - 1, UU)
res = res.vector()
PETSc.Sys.Print(max(abs(max(res)), abs(min(res)))) #l-infinity
test = interpolate(Constant(1), UU)
res = errornorm((1 - 0.25 * inner(phi.dx(0), phi.dx(0))) * inner(phi.dx(1), phi.dx(1)), test, 'l2')
PETSc.Sys.Print(res) #l2


#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

#writing a file with points
points = open('points_%i.txt' % size_ref, 'w')
#if COMM_WORLD.rank == 0:
for i in vec_phi_aux:
  points.write('%.5e %.5e %.5e\n' % (i[0], i[1], i[2]))
points.close()
#sys.exit()

#see if solves other equation
p_aux = 1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))**2
q_aux = 4
test = inner(p_aux * phi.dx(0).dx(0) + q_aux * phi.dx(1).dx(1), div(grad(psi))) * dx
test = assemble(test)
test = project(test, U)
file = File('verif_%i.pvd' % size_ref)
file.write(test)

##other test
#test = interpolate(p(phi) * q(phi), UU)
#with test.dat.vec_ro as v:
#    test = v.max()[1] / 4 * 100
#PETSc.Sys.Print('Max error in percent: %.2e' % test) 

#Citations.print_at_exit()

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
