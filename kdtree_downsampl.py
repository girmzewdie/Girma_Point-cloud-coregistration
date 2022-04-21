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
input_path="C:/M_Geoinformatics/Tutorial/KDtree_normal/"
output_path="C:/M_Geoinformatics/Tutorial/KDtree_normal/"

#dataname="NZ19_Wellington"
#point_cloud=lp.read(input_path+dataname)

pcd = o3d.io.read_point_cloud(input_path+"column_5000pts.pcd")

#print(np.asarray(pcd.points))

# Vizualize
#o3d.visualization.draw_geometries([pcd])

# Voxel downsampling Downsample the point cloud with a voxel of 0.05

downpcd = pcd.voxel_down_sample(voxel_size=0.03)
# downpcd=pcd
# =============================================================================
# o3d.visualization.draw_geometries([downpcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])
# =============================================================================

# Vertex normal estimation
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))

o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)


# Use the below method if the search parameter uses ony the nearest neighborhood
# downpcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
# Access estimated vertex normal, Estimated normal vectors can be retrieved from the normals variable of downpcd.
print(downpcd.normals[0:10])   

#Normal vectors can be transformed as a numpy array using np.asarray.
#print("Print the normal vectors of the first 10 points")
print(np.asarray(downpcd.normals).shape)

# concatenate the downsampled points with the computed normals
pcd_normals=np.concatenate((np.asarray(downpcd.points), np.asarray(downpcd.normals)), axis=1)

# save the downsampled point cloud with the normals
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(downpcd.points))
o3d.io.write_point_cloud(output_path+'column_dwcd.pcd', pcd)

# save the array of downsampled points and normals for further use
np.save(output_path+ 'downpcd_norm.npy', pcd_normals, allow_pickle=True)






