#coding: utf-8
#source firedrake/bin/activate

from firedrake import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys

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
rho = sqrt(4*cos(theta/2)**2*x[0]*x[0] + 1)
phi_D = as_vector((rho*cos(alpha*x[1]), rho*sin(alpha*x[1]), z))

phi_old = Function(V)
#phi_old.project(phi_D)

# Define variational problem for Picard iteration
phi = Function(V, name='solution')
#phi.project(phi_D)
lin_rho = (sqrt(4*cos(theta/2)**2*l*l + 1) - 1) / l * x[0] + 1
phi.project(as_vector((lin_rho*cos(alpha*x[1]), lin_rho*sin(alpha*x[1]), z)))

#bilinear form for linearization
phi_t = TrialFunction(V)
psi = TestFunction(V)
a = inner(p(phi) * phi_t.dx(0).dx(0) + q(phi)*phi_t.dx(1).dx(1), div(grad(psi))) * dx

#penalty to impose Dirichlet BC
h = CellDiameter(mesh)
pen = 1e2
pen_term = pen/h**4 * inner(phi_t, psi) * (ds(1) + ds(2)) #ds
a += pen_term
L = pen/h**4 * inner(phi_D, psi)  * (ds(1) + ds(2)) #ds

##penalty to impose Neumann BC
#n = FacetNormal(mesh)
#pen = 1e1
#pen_term = pen/h/h * inner(dot(grad(phi_t),n), dot(grad(psi),n)) * ds(1) #(ds(1) + ds(2))
#a += pen_term
#L += pen/h/h * inner(dot(grad(phi_D),n), dot(grad(psi),n)) * ds(1) #(ds(1) + ds(2))

#penalty for inequality constraints
pen = 1e2
##pen_ineq = ppos(norm(phi.dx(0)) - sqrt(3))**2 / C * dx
##pen_ineq = pen * ppos(sq_norm(phi.dx(0)) - 3) * dx
pen_ineq = pen * 0.5*(sign(1 - sq_norm(phi.dx(1)))+1) * inner(phi_t.dx(1), psi.dx(1)) * dx
a += pen_ineq
#pen_ineq = pen * 0.5*(sign(sq_norm(phi.dx(0)) - 3)+1) * inner(phi_t.dx(0), psi.dx(0)) * dx
#a += pen_ineq
#pen_ineq = pen * 0.5*(sign(sq_norm(phi.dx(1)) - 4)+1) * inner(phi_t.dx(1), psi.dx(1)) * dx
#a += pen_ineq
##pen_ineq = derivative(pen_ineq, phi, psi)
##pen_ineq = replace(pen_ineq, {phi:phi_t})
##a += lhs(pen_ineq)
##L += rhs(pen_ineq)

##penalty to impose local basis
#pen = 1e2
#pen_ineq = pen * abs(sign(inner(phi.dx(0), phi.dx(1)))) * (inner(phi_t.dx(0), psi.dx(1)) + inner(phi_t.dx(1), psi.dx(0))) * dx
#a += pen_ineq


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


#For projection
U = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
projected = project(phi, U, name='surface')

#Write 2d results
file = File('res.pvd')
file.write(projected)

#check ineq constraints
U = FunctionSpace(mesh, 'CG', 2)
ineq_1 = 0.5*(1+sign(sq_norm(phi.dx(0)) - 3))
ineq_2 = 0.5*(1+sign(sq_norm(phi.dx(1)) - 4))
ineq_3 = 0.5*(1+sign(1 - sq_norm(phi.dx(1))))
constraint = ineq_1 + ineq_2 + ineq_3
constraint = project(constraint, U, name='constraint')
file = File('constraint.pvd')
file.write(constraint)


#check eq constraint
C = CellVolume(mesh)
value = inner(phi.dx(0), phi.dx(1)) / C * dx
print(assemble(value))
#sys.exit()


#plotting solution
vec = projected.vector().get_local()
vec_phi_aux = vec.reshape((len(vec) // 3, 3))

#Nice 3d plot
x = vec_phi_aux[:,0]
y = vec_phi_aux[:,1]
z = vec_phi_aux[:,2]
ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
plt.title('Miura ori')
plt.savefig('miura.pdf')
plt.show()
sys.exit()

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

