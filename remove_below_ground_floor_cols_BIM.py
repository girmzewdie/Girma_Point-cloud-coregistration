# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:35:37 2022

@author: User
"""

import numpy as np
import open3d as o3d
import pandas as pd
import os, glob, pickle
import matplotlib.pyplot as plt

# path of the clusteredcolumns
pc_BIM="C:/M_Geoinformatics/point -cloud-registration/BIM/Coulmn_point cloud.txt"

# read the point cloud in pandas
pc = pd.read_csv(pc_BIM, header=None, delim_whitespace=True).values

# get the coordinates of the point cloud
coord_pc=pc[:,:3]

# remove the negative Z-values
new_coord_pc=[]
# new_coord_pc=[np. zeros([coord_pc.shape[0], coord_pc.shape[1]])]
for i in range(coord_pc.shape[0]):
    if coord_pc[i,2]>=0:
        new_coord_pc.append(coord_pc[i])
        
    else:
        continue
        
# convert the centeroid data into point cloud
arr_new_coord_pc=np.array(new_coord_pc)

# convert point cloud from numpy to ply
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(arr_new_coord_pc)


# save 
o3d.io.write_point_cloud("C:/M_Geoinformatics/point -cloud-registration/BIM/column_pc_ElvationAboveZero.ply", pcd)

# read
points="C:/M_Geoinformatics/point -cloud-registration/BIM/column_pc_ElvationAboveZero.ply"
#points="Area_3/conferenceRoom_1/conferenceRoom_1.ply"
pcd = o3d.io.read_point_cloud(points)

# modify for the reflection in the Bim_columns
trans_init=np.asarray([[-1.0,0.0,0.0,0.0],[0.0,-1.0,0.0,0.0],[0.0,0.0,-1.0,0.0],[0.0,0.0,0.0,1.0]])
pcd.transform(trans_init)

# Vizualize
o3d.visualization.draw_geometries([pcd])

# save the transformed points
o3d.io.write_point_cloud("C:/M_Geoinformatics/point -cloud-registration/BIM/Pre_processed_BIM_pointcloud/column_pc_ElvationAboveZero.ply", pcd)