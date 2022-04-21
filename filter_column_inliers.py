# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:06:35 2022

@author: Girma

This code is for the calculation of the point cloud centeroid 
"""

import numpy as np
import open3d as o3d
import pandas as pd
import os, glob, pickle
import matplotlib.pyplot as plt

# path of the clusteredcolumns
column_cluster_path="C:/M_Geoinformatics/point -cloud-registration/TLS/clustered_columns"

#path for the target_inlier columns
inlier_column_centeroid_path="C:/M_Geoinformatics/point -cloud-registration/TLS/column_centeroid_inliers/target_centeroid_inlier_try.ply"

# read the inlier columns centeroid
target_inliers_load = o3d.io.read_point_cloud("TLS/column_centeroid_inliers/target_centeroid_inlier_try.ply")

# a function for  point cloud to array
def pc2array(pointcloud):
    return np.asarray(pointcloud.points)

def centeroid(arr):
    """A function to calculate the centeroid of each column cluster
    Parram arr: an array of the each column cluster point cloud
    returns: the centeroid of a column cluster"""
    
    cen_x=np.mean(arr[:,0])
    cen_y=np.mean(arr[:, 1])
    cen_z=np.mean(arr[:, 2])

    
  
    # return [sum_x/float(length),sum_y/float(length),sum_z/float(length)]    
    return np.array([cen_x,cen_y, cen_z])

data_list = [] # an empty list to collect all the clusters of the columns

for file in glob.glob(column_cluster_path + '/*.txt'):
     #print(file)
    
     pc = pd.read_csv(file, header=None, delim_whitespace=True).values
     for j in range( pc2array(target_inliers_load).shape[0]):
         # print(centeroid(pc)-pc2array(target_inliers_load)[j])
         diff=(centeroid(pc)-pc2array(target_inliers_load)[j])
         
         if ((diff[0]==0) & (diff[1]==0) & (diff[2]==0)):
     # xc,yc,zc=np.mean(pc[:,0]),np.mean(pc[:,1]),np.mean(pc[:,2]);
     
         # print(type(pc))
      
            data_list.append(pc)
         else:
            continue

# loop over the loop to get all the clusters of column 
for count, column in enumerate(data_list):
    cluster_points=column[:,:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_points)
    o3d.io.write_point_cloud( "TLS/column_cluster_inliers/TLS_column_inliers_column" +str(count )+".ply" , pcd)
    
    
 

# read
column_inliers_merged="C:/M_Geoinformatics/point -cloud-registration/TLS/column_cluster_inliers/TLS_column_inliers_merged.ply"
#points="Area_3/conferenceRoom_1/conferenceRoom_1.ply"
pcd = o3d.io.read_point_cloud(column_inliers_merged)

# Vizualize
o3d.visualization.draw_geometries([pcd])
