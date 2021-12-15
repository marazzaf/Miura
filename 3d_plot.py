#coding: utf-8

import numpy as np
import sys
import open3d as o3d

size_ref = 5
data = np.loadtxt('hyperboloid_%i.txt' % size_ref)

x = data[:,0]
y = data[:,1]
z = data[:,2]

#test new plot
points = np.vstack((x, y, z)).transpose()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
#o3d.visualization.draw_geometries([pcd])

#creating a mesh
#msh = o3d.geometry.TriangleMesh()
#for p in data:
#    msh.vertices.append(p) #adding vertices to the mesh
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
print(mesh)
##add the normals now
#normals = np.loadtxt('normals_%i.txt' % size_ref)
#for n in normals:
#    msh.vertex_normals.append(n) #adding vertices to the mesh
#pcd = msh.sample_points_poisson_disk(3000)

#try to get triangular surface
radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
#o3d.visualization.draw_geometries([pcd, rec_mesh])


