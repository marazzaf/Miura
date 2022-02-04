#coding: utf-8

import numpy as np
import sys
import open3d as o3d

size_ref = 20
data = np.loadtxt('points_%i.txt' % size_ref)

x = data[:,0]
y = data[:,1]
z = data[:,2]

#test new plot
points = np.vstack((x, y, z)).transpose()
pcd = o3d.geometry.PointCloud()
pcd.paint_uniform_color([1, 0.706, 0])
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
