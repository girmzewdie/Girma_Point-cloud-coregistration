# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:06:35 2022

@author: Girma

This code is for the calculation of the point cloud centeroid in 2D plane
"""

import numpy as np
import open3d as o3d
import pandas as pd
import os, glob, pickle
import matplotlib.pyplot as plt

# path of the clusteredcolumns
column_cluster_path="C:/M_Geoinformatics/point -cloud-registration/TLS/clustered_columns/grndstry_column_clusters"

data_list = [] # an empty list to collect all the clusters of the columns
for file in glob.glob(column_cluster_path + '/*.txt'):
     #print(file)
    
     pc = pd.read_csv(file, header=None, delim_whitespace=True).values
     data_list.append(pc)


def centeroid(arr):
    """A function to calculate the centeroid of each column cluster
    Parram arr: an array of the each column cluster point cloud
    returns: the centeroid of a column cluster"""
    length=arr.shape[0]
    sum_x=np.sum(arr[:,0])
    sum_y=np.sum(arr[:, 1])
    sum_z=np.sum(arr[:,2])
    return [sum_x/float(length),sum_y/float(length),sum_z/float(length)]    
    #return np.array([sum_x,sum_y, sum_z])


list_centeroid=[]  # An empty list to collect all the centeroids of each column
for column in data_list:
    #print(len(column))
    Cols_centeroid=centeroid(column)   
    list_centeroid.append(Cols_centeroid)
     
# convert the centeroid data into point cloud
arr_centeroids=np.array(list_centeroid)

"""
x=arr_centeroids[:,0]
y=arr_centeroids[:,1]

# PLot

plt.plot(x,y) 

# Add Titl

plt.title("Matplotlib PLot NumPy Array") 

# Add Axes Labels

plt.xlabel("x axis") 
plt.ylabel("y axis") 

# Display

plt.show()

"""
# convert point cloud from numpy to ply
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(arr_centeroids)
o3d.io.write_point_cloud("C:/M_Geoinformatics/point -cloud-registration/TLS/clustered_columns/col_cent_grnd.ply", pcd)

# read the saved points
points="C:/M_Geoinformatics/point -cloud-registration/TLS/clustered_columns/col_cent_grnd.ply"
pcd = o3d.io.read_point_cloud(points)

# Vizualize
o3d.visualization.draw_geometries([pcd])