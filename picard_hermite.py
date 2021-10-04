from firedrake import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys

# the coefficient functions
def p(phi):
  #return  inner(phi.dx(0), phi.dx(0))**2
  return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  return 4 / inner(phi.dx(1), phi.dx(1))

# Create mesh and define function space
theta = pi/2
L = 2*sin(0.5*acos(0.5/cos(0.5*theta)))
alpha = sqrt(1 / (1 - sin(theta/2)**2))
l = 2*pi/alpha
size_ref = 10 #10 #degub: 5
Nx,Ny = int(size_ref*l/float(L)),size_ref
mesh = RectangleMesh(Nx, Ny, L, l, diagonal="crossed")
V = VectorFunctionSpace(mesh, "HER", 3, dim=3)

# initial guess (its boundary values specify the Dirichlet boundary conditions)
x = SpatialCoordinate(mesh)
z = 2*sin(theta/2)*x[0]
rho = sqrt(4*pow(cos(theta/2),2)*x[0]*x[0] + 1)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

phi_old = Function(V)
#phi_old.project(phi_D)

# Define variational problem for Picard iteration
phi = Function(V, name='solution')
phi.project(phi_D)
phi_t = TrialFunction(V)
psi = TestFunction(V)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx #test
h = CellDiameter(mesh)
pen = 1
a += pen * inner(phi_t, psi)  * ds #/h
L = pen * inner(phi_D, psi)  * ds #/h

# Picard iteration
tol = 1.0E-3
maxiter = 50
for iter in range(maxiter):
  #linear solve
  solve(a == L, phi) # compute next Picard iterate

  ##plotting solution
  #vec_phi_ref = phi.vector().get_local()
  #vec_phi = vec_phi_ref.reshape((3, len(vec_phi_ref) // 3))
  #vec_phi_aux = vec_phi_ref.reshape((len(vec_phi_ref) // 3, 3))
  #
  ##3d plot
  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')
  #for i in vec_phi_aux:
  #  ax.scatter(i[0], i[1], i[2], color='r')
  #plt.show()
  #sys.exit()
    
  eps = sqrt(assemble(inner(grad(phi-phi_old),grad(phi-phi_old))*dx)) # check increment size as convergence test
  #area = assemble(sqrt(1+inner(grad(u),grad(u)))*dx)
  print('iteration{:3d}  H1 seminorm of delta: {:10.2e}'.format(iter+1, eps))
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
file = File('res.pvd')
file.write(projected)

#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

#reference
ref = project(phi_D, U, name='ref')
vec_ref = ref.vector().get_local()
vec_ref = vec.reshape((len(vec_ref) // 3, 3))

#magnitude diff
#img = plot(sqrt(dot(projected-ref, projected-ref)))
#plt.colorbar(img)
#plt.show()
diff = Function(U, name='diff')
diff.vector()[:] = projected.vector() - ref.vector()
file_bis = File('diff.pvd')
file_bis.write(diff)
sys.exit()


#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#for i,j in zip(vec_phi_aux,vec_ref):
for i in vec_phi_aux:
  ax.scatter(i[0], i[1], i[2], color='r')
  #ax.scatter(j[0], j[1], j[2], color='b')
plt.show()

