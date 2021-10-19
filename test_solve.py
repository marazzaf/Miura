#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys
import numpy as np

# the coefficient functions
def p(phi):
  return  inner(phi.dx(0), phi.dx(0))**2
  #return  1 / (1 - 0.25 * inner(phi.dx(0), phi.dx(0)))

def q(phi):
  return 4
  #return 4 / inner(phi.dx(1), phi.dx(1))

def sq_norm(f):
  return inner(f, f)

# Create mesh and define function space
theta = np.pi/2
##check the sizes of the mesh. Might be the problem with some constraints.
L = 2*np.sin(0.5*np.arccos(0.5/np.cos(0.5*theta))) #length of rectangle
#print(L)
alpha = np.sqrt(1 / (1 - np.sin(theta/2)**2))
H = 2*np.pi/alpha #height of rectangle
#print(H)

#Loading mesh
num_computation = 2
mesh = Mesh('rectangle_%i.msh' % num_computation) #change mesh to not use the symmetry any longer
V = VectorFunctionSpace(mesh, "HER", 3, dim=3)

#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)

# initial guess (its boundary values specify the Dirichlet boundary conditions)
x = SpatialCoordinate(mesh)
z = 2*sin(theta/2)*x[0]
rho = sqrt(4*cos(theta/2)**2*x[0]*x[0] + 1)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

phi_old = Function(V)

# Define variational problem for Picard iteration
phi = Function(V, name='solution')
lin_rho = sqrt(4*cos(theta/2)**2*H*H + 1)
phi.project(as_vector((lin_rho*cos(alpha*x[1]), lin_rho*sin(alpha*x[1]), z))) #initial guess is a cylinder

##plotting initial guess
#vec = project(phi, U).vector().get_local()
#vec_phi_aux = vec.reshape((len(vec) // 3, 3))
#
##3d plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for i in vec_phi_aux:
#  ax.scatter(i[0], i[1], i[2], color='r')
#plt.show()
#sys.exit()

#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e1
pen_term = pen/h**4 * inner(phi_t, psi) * (ds(1) + ds(3))
a += pen_term
L = pen/h**4 * inner(phi_D, psi)  * (ds(1) + ds(3))

#penalty for inequality constraints
pen = 1
##pen_ineq = ppos(norm(phi.dx(0)) - sqrt(3))**2 / C * dx
##pen_ineq = pen * ppos(sq_norm(phi.dx(0)) - 3) * dx
pen_ineq = pen * 0.5*(sign(1 - sq_norm(phi.dx(0)))+1) * inner(phi_t.dx(1), psi.dx(1)) * dx
a += pen_ineq
##pen_ineq = derivative(pen_ineq, phi, psi)
##pen_ineq = replace(pen_ineq, {phi:phi_t})
##a += lhs(pen_ineq)
##L += rhs(pen_ineq)

# Picard iteration
tol = 1e-5 #1e-9
maxiter = 50
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


#For plot
projected = project(phi, U, name='surface')

#Write 2d results
file = File('res_%i.pvd' % num_computation)
file.write(projected)

#check ineq constraints
W = FunctionSpace(mesh, 'CG', 2)
ineq_1 = 0.5*(1+sign(sq_norm(phi.dx(0)) - 3))
ineq_2 = 0.5*(1+sign(sq_norm(phi.dx(1)) - 4))
ineq_3 = 0.5*(1+sign(1 - sq_norm(phi.dx(1))))
constraint = ineq_1 + ineq_2 + ineq_3
constraint = project(constraint, W, name='constraint')
file = File('constraint_%i.pvd' % num_computation)
file.write(constraint)


#check eq constraint
C = CellVolume(mesh)
value = inner(phi.dx(0), phi.dx(1)) / C * dx
print(assemble(value))
#sys.exit()


#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in vec_phi_aux:
  ax.scatter(i[0], i[1], i[2], color='r')
#plt.show()
plt.title('Miura ori')
plt.savefig('miura_bis_%i.pdf' % num_computation)

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

#magnitude diff
#img = plot(sqrt(dot(projected-ref, projected-ref)))
#plt.colorbar(img)
#plt.show()
diff = Function(U, name='diff')
diff.vector()[:] = projected.vector() - ref.vector()
file_bis = File('diff_%i.pvd' % num_computation)
file_bis.write(diff)

