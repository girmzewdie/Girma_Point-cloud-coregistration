# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 07:53:05 2022

@author: User
"""
import open3d as o3d
import numpy as np
from matplotlib import pyplot as pyplot
import scipy
import copy
from scipy import spatial 
import time
import random
import os
import sys
import math

# read source columns
# source_pc="C:/M_Geoinformatics/point -cloud-registration/BIM/column_centeroid/BIM_column_centeroid_3D/BIM_column_centeroid_3D -mod.ply"
source_pc="C:/M_Geoinformatics/point -cloud-registration/BIM/column_centeroid/BIM_column_centeroid_3D/BIM_column_centeroid_3D.ply"
# source_cols="C:/M_Geoinformatics/point -cloud-registration/BIM/BIM_filtered_Coulmn_point cloud _grnd.ply"
source_cols="C:/M_Geoinformatics/point -cloud-registration/BIM/Pre_processed_BIM_pointcloud/groundFloor_columns_ElvationAboveZero_reflect.ply"

# read target columns
# target_pc="C:/M_Geoinformatics/point -cloud-registration/TLS/column_centeroid/3D/TLS_column_centeroid_3D _mod -248.ply"
target_pc="C:/M_Geoinformatics/point -cloud-registration/TLS/column_centeroid/3D/TLS_column_centeroid_3D.ply"
# target_cols="C:/M_Geoinformatics/point -cloud-registration/TLS/filtered_columns/TLS_column_grnd_uncleaned_ckpt5.ply"
target_cols="C:/M_Geoinformatics/point -cloud-registration/TLS/filtered_columns/TLS_column_grnd_cleaned_ckpt5.ply"

# read column centeroids for both dataset
source=o3d.io.read_point_cloud(source_pc)
target=o3d.io.read_point_cloud(target_pc) 

# read filtered columns for both dataset
source_cols=o3d.io.read_point_cloud(source_cols)
target_cols=o3d.io.read_point_cloud(target_cols) 

# # modify for the reflection in the Bim_columns
# trans_init=np.asarray([[-1.0,0.0,0.0,0.0],[0.0,-1.0,0.0,0.0],[0.0,0.0,-1.0,0.0],[0.0,0.0,0.0,1.0]])
# source_cols.transform(trans_init)

# Vizualize
o3d.visualization.draw_geometries([source_cols]) 

# point cloud to array
def pc2array(pointcloud):
    return np.asarray(pointcloud.points)






# 3D rotational matrix along Z direction

def Rz(theta):
  return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                    [ math.sin(theta), math.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])


        
#compute the transformed points from source to target based on the R/T found in PCA Algorithm
def _transform(target,R,T):
    points = []
    for point in target:
        points.append(np.dot(R,point.reshape(-1,1))+T)
    return np.array(points)


"""
rmse = 0
number = pc2array(source).shape[0]
points = _transform(source,R,T)
for i in range(number):
    error = pc2array(target)[i].reshape(-1,1)-np.array(points)[i].reshape(-1,1)
    rmse = rmse + math.sqrt(error[0]**2+error[1]**2+error[2]**2)
print(rmse)
"""
#compute the distance between the each column enteroids of the source wirth the transformed target column centeroid
def compute_distance_2D(source, target,R,T):
    transform_target =_transform(target,R,T)
    
    dist_source_target=np.zeros((transform_target.shape[0],source.shape[0]))  
    for i in range (transform_target.shape[0]):
        for j in range(source.shape[0]):
            calc_dist=math.sqrt((transform_target[i,0]-source[j,0])**2+(transform_target[i,1]-source[j,1])**2)
            dist_source_target[i,j]=calc_dist
    return dist_source_target
    
            
                   
 
                  
def compute_rmse_2D(source,target,R,T):
    rmse = 0
    distance=[]
    ditance_all=[]
    number = target.shape[0]
    points = _transform(target,R,T)
    dist_diff=compute_distance_2D(source,target,R,T)
    for i in range(number):
        # error = source[i].reshape(-1,1)-np.array(points)[i].reshape(-1,1)        
        dist=np.min(dist_diff[i])
        ditance_all.append(dist)
        # error=np.abs(source[i].reshape(-1,1)-np.array(points)[i].reshape(-1,1))
        if dist<=0.5:      
            distance.append(dist)
            rmse = rmse + dist
        else:
            continue
        
    return distance,rmse,ditance_all




# Visualization the transformation result
def draw_registrations(source, target, transformation = None, recolor = False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if(recolor): # recolor the points
        source_temp.paint_uniform_color([1, 0.706, 0])

        target_temp.paint_uniform_color([0, 0.651, 0.929])
        
    if(transformation is not None): # transforma target to source
        target_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# vizualize original pc
# trans_init=np.asarray([[-1.0,0.0,0.0,0.0],[0.0,-1.0,0.0,0.0],[0.0,0.0,-1.0,0.0],[0.0,0.0,0.0,1.0]])
# source.transform(trans_init)
draw_registrations(source, target, np.identity(4),True)

     
############################### R A N S A C ############################################     
   
def registration_RANSAC(source,target,ransac_n,max_iteration,threshold_dist,threshold_inlier_ratio):
    best_model = None
    best_inliers = None
    best_num_inliers = 0
    opt_rmse = np.inf
    inlier_ratio=0
    all_dataset=0
    
    #the intention of RANSAC is to get the optimal transformation between the source and target point cloud
    s=pc2array(source)[:,:2]
    t=pc2array(target)[:,:2]    
    row,col=pc2array(target)[:,:2] .shape
    number_col=pc2array(target).shape[0]
    # inliers
    
    
    # create a look up table which contains the distance between each column in the target poinyt cloud

    lookup_dist=np.zeros((row,row))  
    for i in range(row):      
        for j in range(row):
            calc_dist=math.sqrt(((t[i,0]-t[j,0]))**2+((t[i,1]-t[j,1]))**2)
            lookup_dist[i,j]=calc_dist                     
        
        
    # print(lookup_dist.shape)
    # print(10*"*")
    # print(lookup_dist)
    
    # corresponding columns in target search function
    def search_corres_column(distance_dicr, distance_lookup):
        row,col=t.shape
        for i in range(row):
            corr_set=[]
            for j in range(row):
                if ((distance_dicr-0.5) <= (distance_lookup[i,j]) <= (distance_dicr+0.5)):
                    # target_corr=[i,j]
                    corr_set.append([i,j])
                else:
                    continue
        return corr_set
            
    
    for k in range(max_iteration):
        
        #take ransac_n points randomly
    
        idx_s = np.random.choice(np.arange(s.shape[0]), ransac_n, replace=False)
        source_point = s[idx_s,...]
        ds=math.sqrt(((source_point[0,0]-source_point[1,0]))**2+((source_point[0,1]-source_point[1,1]))**2)
                
         # select corresponding columns from target point with constraint
         
        corr_set=search_corres_column(ds, lookup_dist)
        corr_set_arr=np.asarray(corr_set)
        if inlier_ratio > threshold_inlier_ratio:
        # if opt_rmse < 0.30 and inlier_ratio > threshold_inlier_ratio :
            break 
        
        
        # loop over the selected target point correspondences
                       
        if corr_set_arr.size > 0:
            for ii in range(corr_set_arr.shape[0]):                            
                idx_t = corr_set[ii]
                target_point = t[idx_t,...]
        
                # compute transformation
                q=source_point
                p=target_point
                ####################
                q1x= source_point[0][0]
                q1y=source_point[0][1]
                
                q2x=source_point[1][0]
                q2y=source_point[1][1]
                ######################
                p1x=target_point[0][0]
                p1y=target_point[0][1]
                
                p2x=target_point[1][0]
                p2y=target_point[1][1]
                
               
                # compute theta from the literature
                sy=q2y-q1y
                tx=p2x-p1x
                sx=q2x-q1x
                ty=p2y-p1y      
                    
                x=((sy)*(tx/sx))-ty
                y=((sy)*(ty/sx))+tx        
                #theta
                theta=math.atan(x/y)
                
                # theta=0 
                
                # construct the rotation matrix                
                R_2D=np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
                
                # R_init=np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
                # deg=math.degrees(theta)              
                             
                # 2D translation
                Tx=q1x-(p1x*math.cos(theta)) +(p1y*math.sin(theta))
                Ty=q1y-(p1x*math.sin(theta))-(p1y*math.cos(theta))                      

                T_2D=np.array([[Tx,Ty]]).reshape(-1,1) 
                """
                
                
                # Using the least squares library from numpy
                
                # create A matrix containing known coordinates of the target point
                A=np.asarray([[p1x,-p1y,1.0,0.0],[p1y,p1x,0.0,1.0],
                                   [p2x,-p2y,1.0,0.0],[p2y,p2x,0.0,1.0]])
        
                # b matrix containing known coordinates of the source points
                B=np.asarray([q1x,q1y,q2x,q2y]).reshape(-1,1)
                # compute the inverese
                # if det(A) !=0:
                #     A_inv=np.linalg.inv(A)
                # # compute the unknown sinilarity transformation parameters  
                
                # x=np.dot(A_inv,B)
                   
                # by least squares
                lst_x,res,r,ss=np.linalg.lstsq(A, B, rcond=None)       
                
                
                # the four unknown parameters
                a=lst_x[0]; b=lst_x[1]
                # compute theta: s*cos(theta)=a; s*sin(theta)=b
                theta=math.atan(b/a)
                
                                
                # slope
                slope= math.sqrt(a**2+b**2)   
                
                # Rotation matrix  
                R_2D=np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
                
                # translation vextor
                # Tx=lst_x[2]; Ty=lst_x[3]
                # 2D translation
                Tx=q1x-(p1x*math.cos(theta)) +(p1y*math.sin(theta))
                Ty=q1y-(p1x*math.sin(theta))-(p1y*math.cos(theta))   

                T_2D=np.array([[Tx,Ty]]).reshape(-1,1)
                
                # R_2D,T_2D = compute_2D_transformation(source_point,target_point)
                """
                
                print(corr_set_arr,idx_t,idx_s)
                
                #calculate rmse for all points
                dist1,rmse1,distance_all=compute_rmse_2D(s,t,R_2D,T_2D)     
                error_arr=np.array(dist1)
                error_all=np.array(distance_all)
                # print(error_arr)
                centers=error_arr.shape
                
             
                inliers = (error_arr<=threshold_dist)               
                num_inliers = np.sum(inliers)
                inlier_ratio=(num_inliers/number_col)
                
                print(opt_rmse,best_num_inliers)
                # compute the sum of all inliers and outliers
                all_dataset=(error_all<=threshold_dist) 
                      
                        
                # if not k:
                #     best_num_inliers = num_inliers
                #     opt_rmse = rmse1 
                #     best_inliers = inliers
                #     inlier_ratio=(best_num_inliers/number_col)
                #     opt_R = R_2D
                #     opt_T = T_2D
                #     opt_dist=error_arr
                    
                # else:
                if (best_num_inliers < num_inliers): 
                # if (rmse1 < opt_rmse) and (best_num_inliers < num_inliers):
                # if (rmse1 < opt_rmse):
                    best_num_inliers = num_inliers
                    opt_rmse = rmse1 
                    best_inliers = inliers
                    inlier_ratio=(best_num_inliers/number_col)
                    opt_dist=error_arr
                    opt_R = R_2D
                    opt_T = T_2D
                    bothInliersAndOutliers=all_dataset
                     
            
        else:
            continue
        
        
    
                        
       
    print("the number of iteration taken is " + str(k))
                    
    return opt_rmse,opt_R, opt_T,best_inliers, best_num_inliers,inlier_ratio,opt_dist,bothInliersAndOutliers

start = time.time()
opt_rmse,opt_R, opt_T,best_inliers, best_num_inliers,inlier_ratio,opt_dist,bothInliersAndOutliers= registration_RANSAC(source,target,ransac_n=2,max_iteration=1000, threshold_dist=0.5,threshold_inlier_ratio=0.97)

print("Global registration took %.3f sec.\n" % (time.time() - start))
            

opt_theta=math.acos(opt_R[:,0][0])


# 3D rotation matrix 
def Rz(theta):
  return np.matrix([[ math.cos(theta), math.sin(theta), 0 ],
                    [ -math.sin(theta), math.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])

# 3D_translation_vector


# 3D translation vector, Tx & Ty are from the ransac and z=0 result

translation_z=np.matrix([[np.float64(opt_T[0]), np.float64(opt_T[1]),np.float64(0)]]).reshape(-1,1)

# 3D similarity transformation matrix, assuming scale =1 and tz=0
transformation = np.vstack((np.hstack((np.float64(Rz(opt_theta)), np.float64(translation_z))), np.array([0,0,0,1])))







"""
def compute_Z_translation(source,target):
        # compute translation along the Z-axis by using the transformed target points

        #  returns the 3D translation vector
        
    
    # transformed_target=target.transform(transformation)
    source_arr=pc2array(source)
    target_arr=pc2array(target)
    
    #the centroid of source points
    cs = np.zeros((3,1))
    #the centroid of target points
    ct = copy.deepcopy(cs)    
    
    cs[0] = np.mean(source_arr[:,0]);cs[1]=np.mean(source_arr[:,1]);cs[2]=np.mean(source_arr[:,2])
    ct[0] = np.mean(target_arr[:,0]);cs[1]=np.mean(target_arr[:,1]);cs[2]=np.mean(target_arr[:,2])    
   
    #Translation vector
    T= cs - np.dot(Rz(opt_theta),ct)
    return T
"""
"""
def compute_translation_3D(source,target):
    

    number=source.shape[0]
    
    #the centroid of source points
    cs = np.zeros((3,1))
    #the centroid of target points
    ct = copy.deepcopy(cs)    
    
    cs[0] = np.mean(source[:,0]);cs[1]=np.mean(source[:,1]);cs[2]=np.mean(source[:,2])
    ct[0] = np.mean(target[:,0]);ct[1]=np.mean(target[:,1]);ct[2]=np.mean(target[:,2])
    #covariance matrix
    cov_s = np.zeros((3,3))
    cov_t = np.zeros((3,3))
    #translate the centroids of both models to the origin of the coordinate system (0,0,0)
    #subtract from each point coordinates the coordinates of its corresponding centroid
    for i in range(number):
        sourc_red = source[i].reshape(-1,1)-cs
        targ_red = target[i].reshape(-1,1)-ct
        cov_s = cov_s + np.dot(sourc_red,np.transpose(sourc_red))
        cov_t= cov_t + np.dot(targ_red,np.transpose(targ_red))
    #SVD (singular values decomposition)
    u_s,w_s,v_s = np.linalg.svd(cov_s)    
    u_t,w_t,v_t = np.linalg.svd(cov_t)
    # invert u_t
    u_t_inv=np.linalg.inv(u_t)
    
    #rotation matrix
    R = np.dot(u_s,u_t_inv)
    # R = np.dot(u,v)
    #Transformation vector
    T = ct - np.dot(R,cs)
    Tin = cs - ct
    return T

T_3D=compute_translation_3D(pc2array(source_cols),pc2array(target_cols))
"""
# 3D translation vector, Tx & Ty are from the resc result
translation_3D=np.matrix([[np.float64(opt_T[0]), np.float64(opt_T[1]),np.float64(0)]]).reshape(-1,1)
# 3D similarity transformation matrix, assuming scale =1
transformation_3D = np.vstack((np.hstack((np.float64(Rz(opt_theta)), np.float64(translation_3D))), np.array([0,0,0,1])))

transformation_param={"Trans_RANS":transformation_3D,"rotation_RANS_2D":opt_R, "Translation_RANS_2D": opt_T,"translation_3D" :transformation_3D, "distance_difference":opt_dist,"number_inliers":best_num_inliers,"opt_rmse_RANS_2D":opt_rmse/257, "iteration_required":647, "time_sec": 1256}

#save
"""
import pickle as pkl
with open('transformations/RANSAC_3D_dist_corr_similartytransfom3.pkl', 'wb') as handle:
    pkl.dump(transformation_param, handle)  

#########################################################################################################                  
#load

with open('transformations/RANSAC_3D_dist_corr_similartytransfom2.pkl', 'rb') as handle:
    b = pkl.load(handle)
trans_RANSAC=b['Trans_RANS']
"""

# vizualize the registration result( column centeroids)
draw_registrations(source,target, transformation_3D, True)

# vizualize the registration result(filtered columns)
draw_registrations(source_cols, target_cols,transformation_3D, True)





# check with the entire point cloud

input_path_source="C:/M_Geoinformatics/point -cloud-registration/BIM/Entire_BIM_point cloud/BIM_full_pc.ply"
input_path_target="C:/M_Geoinformatics/point -cloud-registration/TLS/entire_bldg_pointcloud/TLS_ITC_full_pc.ply"

# read both dataset
pcd_source = o3d.io.read_point_cloud(input_path_source)
pcd_target = o3d.io.read_point_cloud(input_path_target)


"""
# Voxel downsampling Downsample the point cloud with a voxel of 0.07
downpcd_source = pcd_source.voxel_down_sample(voxel_size=0.05)
downpcd_target = pcd_target.voxel_down_sample(voxel_size=0.05)

draw_registrations(downpcd_source, downpcd_target, np.identity(4),True)
"""

# vizualize the registration result
draw_registrations(pcd_source, pcd_target,transformation_3D, True)

# get the point cloud column centeroids corresponding to the inliers
s=pc2array(source)[:,:2]
t=pc2array(target)
target_inliers= t[bothInliersAndOutliers]

# convert to the open3d format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(target_inliers)
o3d.io.write_point_cloud("TLS/column_centeroid_inliers/target_centeroid_inlier_try.ply", pcd)

# visualize
target_inliers_load = o3d.io.read_point_cloud("TLS/column_centeroid_inliers/target_centeroid_inlier_try.ply")
o3d.visualization.draw_geometries([target_inliers_load])

# transform the inliers using the coarse transfomation result from RANSAC
draw_registrations(source,target_inliers_load, transformation_3D, True)


