#coding: utf-8

import numpy as np
import sys
import open3d as o3d

size_ref = 40
#data = np.loadtxt('hyperboloid_%i.txt' % size_ref)
#data = np.loadtxt('points_aux_%i.txt' % size_ref)
data = np.loadtxt('points_%i.txt' % size_ref)

x = data[:,0]
y = data[:,1]
z = data[:,2]

#test new plot
points = np.vstack((x, y, z)).transpose()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
sys.exit()

#creating a mesh
#msh = o3d.geometry.TriangleMesh()
#for p in data:
#    msh.vertices.append(p) #adding vertices to the mesh

#add the normals now
normals = np.loadtxt('normals_%i.txt' % size_ref)
for n in normals:
    pcd.normals.append(n) #adding normals to the point cloud
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
sys.exit()

#Constructing the mesh
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
#o3d.visualization.draw_geometries([mesh])
sys.exit()

#try to get triangular surface
radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])


