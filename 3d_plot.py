#coding: utf-8

import numpy as np

size_ref = 5
data = np.loadtxt('hyperboloid_%i.txt' % size_ref)

x = data[:,0]
y = data[:,1]
z = data[:,2]

#test new plot
points = np.vstack((x, y, z)).transpose()
#colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
#pcd.colors = o3d.utility.Vector3dVector(colors/65535)
#pcd.normals = o3d.utility.Vector3dVector(normals)
o3d.visualization.draw_geometries([pcd])
