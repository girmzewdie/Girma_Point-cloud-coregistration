
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 20:49:30 2021

@author: Girma

link"http://www.open3d.org/docs/latest/tutorial/Basic/kdtree.html"
"""


import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt

#input_path="C:/Trashes/MSc_files/Datasets/forpractice/Point Cloud Sample-20210726T080041Z-001/Point Cloud Sample/"
input_path="C:\M_Geoinformatics\Tutorial\KDtree_normal/"
output_path="C:\M_Geoinformatics\Tutorial\KDtree_normal/"

#dataname="NZ19_Wellington"
#point_cloud=lp.read(input_path+dataname)

pcd = o3d.io.read_point_cloud(input_path+"column_5000pts.pcd")

#print(np.asarray(pcd.points))

# Vizualize
# o3d.visualization.draw_geometries([pcd],
#                                     zoom=0.3412,
#                                     front=[0.4257, -0.2125, -0.8795],
#                                     lookat=[2.6172, 2.0475, 1.532],
#                                     up=[-0.0694, -0.9768, 0.2024])

#o3d.visualization.draw_geometries([pcd])

#Testing kdtree in Open3D
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

#Load a point cloud and paint it gray.

pcd.paint_uniform_color([0.5, 0.5, 0.5])

# Find neighboring points
# We pick the 1500th point as the anchor point and paint it red.
pcd.colors[10] = [1, 0, 0]

# Using search_knn_vector_3d
# =============================================================================
# print("Find its 50 nearest neighbors, and paint them blue.")
# [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[10], 50)
# # 
# np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
# =============================================================================

# Using search_radius_vector_3d
"""Similarly, we can use search_radius_vector_3d to query all points with 
distances to the anchor point less than a given radius. We paint these points
 with a green color.
"""
print("Find its neighbors with distance less than 0.2, and paint them green.")
# [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[10], 0.1)
# np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

# hybrid serach
[k, idx, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[10], 0.1,80)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

# visualize

print("Visualize the point cloud.")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.5599,
                                  front=[-0.4958, 0.8229, 0.2773],
                                  lookat=[2.1126, 1.0163, -1.8543],
                                  up=[0.1007, -0.2626, 0.9596])



#compute normals using the following command
#pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)


