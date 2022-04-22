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
def compute_target_source_distance_2D(source, target,R,T):
    """computes the the corresponding distance between each target column and 
    the source column centeroid.
    Returns the an array(row size: the number of target columns; column: the number of source columns)"""
    
    transform_target =_transform(target,R,T)    
    dist_target_source=np.zeros((transform_target.shape[0],source.shape[0]))    
    for i in range (transform_target.shape[0]):
        for j in range(source.shape[0]):
            calc_dist=math.sqrt((transform_target[i,0]-source[j,0])**2+(transform_target[i,1]-source[j,1])**2)
            dist_target_source[i,j]=calc_dist
    return dist_target_source

def compute_source_target_distance_2D(source, target,R,T):
    """computes the the corresponding distance between each source column and 
    the target column centeroid.
    Returns the an array(row size: the number of source columns; column: the number of target columns)"""
    
    transform_target =_transform(target,R,T)    
    dist_source_target=np.zeros((source.shape[0],transform_target.shape[0]))
    for i in range (source.shape[0]):
        for j in range(transform_target.shape[0]):
            calc_dist=math.sqrt((source[i,0]-transform_target[j,0])**2+(source[i,1]-transform_target[j,1])**2)
            dist_source_target[i,j]=calc_dist
    return dist_source_target    
            
                   
 
                  
def compute_rmse_target_source_2D(source,target,R,T):
    rmse = 0
    distance=[]
    ditance_all=[]
    number = target.shape[0]
    points = _transform(target,R,T)
    dist_diff=compute_target_source_distance_2D(source,target,R,T)
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

def compute_rmse_source_target_2D(source,target,R,T):
    rmse = 0
    distance_source_target_inliers=[]
    ditance_source_target_all=[]
    number = source.shape[0]
    points = _transform(target,R,T)
    dist_diff=compute_source_target_distance_2D(source,target,R,T)
    for i in range(number):
        # error = source[i].reshape(-1,1)-np.array(points)[i].reshape(-1,1)        
        dist=np.min(dist_diff[i])
        ditance_source_target_all.append(dist)
        # error=np.abs(source[i].reshape(-1,1)-np.array(points)[i].reshape(-1,1))
        if dist<=0.5:      
            distance_source_target_inliers.append(dist)
            rmse = rmse + dist
        else:
            continue
        
    return distance_source_target_inliers,rmse,ditance_source_target_all



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

     
############################### R A N S A C  begin coarse registration using RANSAC ############################################     
   
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
        # if opt_rmse < 45 and inlier_ratio > threshold_inlier_ratio :
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
                
                #calculate rmse and distance for all transformed target points 
                dist1,rmse1,distance_all=compute_rmse_target_source_2D(s,t,R_2D,T_2D) 
                dist2,rmse2,distance_all2=compute_rmse_source_target_2D(s,t,R_2D,T_2D)
                
                error_arr=np.array(dist1)
                
                # all the distance between the target columns and the source columns(size==number of target columns)
                error_all_target=np.array(distance_all)    
                
                # all the distance between the source columns and the target columns(size==number of source columns)
                error_all_source=np.array(distance_all2) 
                centers=error_arr.shape[0]
                print(centers)
                
             
                inliers = (error_arr<=threshold_dist)               
                num_inliers = np.sum(inliers)
                inlier_ratio=(num_inliers/number_col)
                
                print(opt_rmse,best_num_inliers)
                # compute the sum of all inliers and outliers
                all_dataset_target=(error_all_target<=threshold_dist) 
                all_dataset_source=(error_all_source<=threshold_dist)
                      
                        
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
                    bothInliersAndOutliers_target=all_dataset_target
                    bothInliersAndOutliers_source=all_dataset_source
                     
            
        else:
            continue
        
        
    
                        
       
    print("the number of iteration taken is " + str(k))
                    
    return opt_rmse,opt_R, opt_T,best_inliers, best_num_inliers,inlier_ratio,opt_dist,bothInliersAndOutliers_target,bothInliersAndOutliers_source

start = time.time()
opt_rmse,opt_R, opt_T,best_inliers, best_num_inliers,inlier_ratio,opt_dist,bothInliersAndOutliers_target,bothInliersAndOutliers_source= registration_RANSAC(source,target,ransac_n=2,max_iteration=1000, threshold_dist=0.5 ,threshold_inlier_ratio=0.9)

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

# 3D translation vector, Tx & Ty are from the resc result, and tz=0
translation_3D=np.matrix([[np.float64(opt_T[0]), np.float64(opt_T[1]),np.float64(0)]]).reshape(-1,1)

# 3D similarity transformation matrix, assuming scale =1
transformation_3D = np.vstack((np.hstack((np.float64(Rz(opt_theta)), np.float64(translation_3D))), np.array([0,0,0,1])))

# store the RANSAC result in the dictionary for sterilization
transformation_param={"Trans_RANS":transformation_3D,"rotation_RANS_2D":opt_R, "Translation_RANS_2D": opt_T,"translation_3D" :transformation_3D, "distance_difference":opt_dist,"number_inliers":best_num_inliers,"opt_rmse_RANS_2D":opt_rmse/best_num_inliers,"bothInliersAndOutliers_target":bothInliersAndOutliers_target,"bothInliersAndOutliers_source":bothInliersAndOutliers_source ,"iteration_required":547, "time_sec": 1256}

#save

import pickle as pkl
with open('transformations/RANSAC_3D_dist_corr_similartytransfom_aprl22.pkl', 'wb') as handle:
    pkl.dump(transformation_param, handle)  

##################################  E N D  Coarse registration using RANSAC completed #######################################################################                  

#load the RANSAC result
with open('transformations/RANSAC_3D_dist_corr_similartytransfom_aprl22.pkl', 'rb') as handle:
    b = pkl.load(handle)

# get the 3D transformation(4x4) matrix from RANSAC
trans_RANSAC=b['Trans_RANS']

# Get the number of inliers, RMSE
numb_best_inliers=b['number_inliers']
print("The number of best inliers are", numb_best_inliers)
rmse_RANSAC= b['opt_rmse_RANS_2D']
print("The RMSE of coarse registration from RANSA is", rmse_RANSAC)



# vizualize the registration result( column centeroids)
draw_registrations(source,target, trans_RANSAC, True)

# vizualize the registration result(filtered columns)
draw_registrations(source_cols, target_cols,trans_RANSAC, True)

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
draw_registrations(pcd_source, pcd_target,trans_RANSAC, True)

# get the inliers of the point cloud column centeroids and the corresponding BIM columns
s=pc2array(source)
t=pc2array(target)

# inliers
inlier_target=b['bothInliersAndOutliers_target']
inlier_source=b['bothInliersAndOutliers_source']
target_inliers= t[inlier_target]
source_corresponding_inliers=s[inlier_source]


# convert target inlier columns into the open3d format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(target_inliers)
o3d.io.write_point_cloud("TLS/column_centeroid_inliers/target_centeroid_inlier_try.ply", pcd)

# visualize
target_inliers_load = o3d.io.read_point_cloud("TLS/column_centeroid_inliers/target_centeroid_inlier_try.ply")
o3d.visualization.draw_geometries([target_inliers_load])



# convert source inlier columns into the open3d format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(source_corresponding_inliers)
o3d.io.write_point_cloud("BIM/column_cluster_inliers/source_centeroid_inlier_try.ply", pcd)

# visualize
source_inliers_load = o3d.io.read_point_cloud("BIM/column_cluster_inliers/source_centeroid_inlier_try.ply")
o3d.visualization.draw_geometries([source_inliers_load])

# transform the inliers using the coarse transfomation result from RANSAC
draw_registrations(source_inliers_load,target_inliers_load, transformation_3D, True)


######################################## P C A  apply PCA on the inlier centeroids ######################################

# apply a PCA on the inliers from the RANSAC

def PCA_compute_2D_transformation(source,target):
    """Computes a 2D transformation result for the redundant corresponding points"""
    #Normalization
    number=source.shape[0]
    
    #the centroid of source points
    cs = np.zeros((2,1))
    #the centroid of target points
    ct = copy.deepcopy(cs)    
    
    cs[0] = np.mean(source[:,0]);cs[1]=np.mean(source[:,1])
    ct[0] = np.mean(target[:,0]);ct[1]=np.mean(target[:,1])
    #covariance matrix
    cov_s = np.zeros((2,2))
    cov_t = np.zeros((2,2))
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
    T = cs - np.dot(R,ct)
    Tin = cs - ct
    return R, T

# call the 2d PCA transformation function
R_PCA, T_PCA = PCA_compute_2D_transformation(pc2array(source_inliers_load)[:,:2],pc2array(target_inliers_load)[:,:2])

# 3D translation vector, Tx & Ty are from the PCA and take the translation along Z-axis as z=0
translation_PCA=np.matrix([[np.float64(T_PCA[0]), np.float64(T_PCA[1]),np.float64(0)]]).reshape(-1,1)

# get the rotation angle along z_axis from the 2D PCA transformation result
PCA_theta=math.acos(R_PCA[:,0][0])

# Construct the 3D similarity transformation matrix, assuming scale =1 and tz=0
transformation_PCA = np.vstack((np.hstack((np.float64(Rz(PCA_theta)), np.float64(translation_PCA))), np.array([0,0,0,1])))

# vizualize the PCA transformation result on the colomn centeroids
draw_registrations(source_inliers_load, target_inliers_load, np.identity(4),True)
draw_registrations(source_inliers_load, target_inliers_load, transformation_PCA, True)

# Compute the RMSE
def _transform(target,R,T):
    """A function to transform the target point cloud
    Returns the trhe transformed point array"""
    points = []
    for point in target:
        points.append(np.dot(R,point.reshape(-1,1))+T)
    return np.array(points)

def compute_target_source_distance_PCA(source, target,R,T):
    """computes the the corresponding distance between each target column and 
    the source column centeroid.
    Returns the an array(row size: the number of target columns; column: the number of source columns)"""
    
    transform_target =_transform(target,R,T)    
    dist_target_source=np.zeros((transform_target.shape[0],source.shape[0]))    
    for i in range (transform_target.shape[0]):
        for j in range(source.shape[0]):
            calc_dist=math.sqrt((transform_target[i,0]-source[j,0])**2+(transform_target[i,1]-source[j,1])**2 )
            dist_target_source[i,j]=calc_dist
    return dist_target_source

def compute_rmse_target_source_PCA(source,target,R,T):
    """computes the RMSE of coarse 2D coarse registration result using PCA"""
    rmse = 0
    distance=[]
    ditance_all=[]
    number = target.shape[0]
    points = _transform(target,R,T)
    dist_diff=compute_target_source_distance_PCA(source,target,R,T)
    for i in range(number):              
        dist=np.min(dist_diff[i])
        ditance_all.append(dist)      
        rmse = rmse + dist        
    return distance,rmse/number,ditance_all

# call a function to compute a coarse registration RMSE on the 2D plane and the distance between each corresponding column centeroids
distance_PCA,rmse_PCA,ditance_all_PCA=compute_rmse_target_source_PCA(pc2array(source_inliers_load)[:,:2],pc2array(target_inliers_load)[:,:2],R_PCA,T_PCA)
print("The RMSE of coarse registration is" , rmse_PCA)

# Visualize the PCA registration result on the columns only without filtering
draw_registrations(source_cols, target_cols,transformation_PCA, True)

# vizualize the PCA registration result on the entire point cloud
draw_registrations(pcd_source, pcd_target,transformation_PCA, True)


################################ Filter inlier columns ###############################################################

# path of the target and source clustered columns
target_column_cluster_path="C:/M_Geoinformatics/point -cloud-registration/TLS/clustered_columns"
source_column_cluster_path="C:/M_Geoinformatics/point -cloud-registration/BIM/clustered_columns"

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


target_inlier_data_list = [] # an empty list to collect all the clusters of the inlier columns

# loop over each column clusters and collect the inliers corresponding to the centeroid inliers
for file in glob.glob(target_column_cluster_path + '/*.txt'):
     #print(file)    
     pc = pd.read_csv(file, header=None, delim_whitespace=True).values
     for j in range( pc2array(target_inliers_load).shape[0]):
         # print(centeroid(pc)-pc2array(target_inliers_load)[j])
         diff=(centeroid(pc)-pc2array(target_inliers_load)[j])         
         if ((diff[0]==0) & (diff[1]==0) & (diff[2]==0)):
     # xc,yc,zc=np.mean(pc[:,0]),np.mean(pc[:,1]),np.mean(pc[:,2]);     
         # print(type(pc))      
            target_inlier_data_list.append(pc)
         else:
            continue

# loop over the loop to get all the clusters of column 
for count, column in enumerate(target_inlier_data_list):
    cluster_points=column[:,:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_points)
    o3d.io.write_point_cloud( "TLS/column_cluster_inliers/TLS_column_inliers_column" +str(count )+".ply" , pcd)
    
source_inlier_data_list = [] # an empty list to collect all the clusters of the inlier columns

# loop over each column clusters and collect the inliers corresponding to the centeroid inliers
for file in glob.glob(source_column_cluster_path + '/*.txt'):
     #print(file)    
     pc = pd.read_csv(file, header=None, delim_whitespace=True).values
     for j in range( pc2array(source_inliers_load).shape[0]):
         # print(centeroid(pc)-pc2array(target_inliers_load)[j])
         diff=(centeroid(pc)-pc2array(source_inliers_load)[j])         
         if ((diff[0]==0) & (diff[1]==0) & (diff[2]==0)):
     # xc,yc,zc=np.mean(pc[:,0]),np.mean(pc[:,1]),np.mean(pc[:,2]);     
         # print(type(pc))      
            source_inlier_data_list.append(pc)
         else:
            continue

# loop over the loop to get all the clusters of column 
for count, column in enumerate(source_inlier_data_list):
    cluster_points=column[:,:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_points)
    o3d.io.write_point_cloud( "BIM/column_cluster_inliers/BIM_column_inliers" +str(count )+".ply" , pcd)    
 

# read the filtered column inliers of both dataset
TLS_column_inliers_merged="C:/M_Geoinformatics/point -cloud-registration/TLS/column_cluster_inliers/TLS_column_inliers_merged.ply"
BIM_column_inliers_merged= "C:/M_Geoinformatics/point -cloud-registration/BIM/column_cluster_inliers/BIM_column_inliers_merged.ply"
#points="Area_3/conferenceRoom_1/conferenceRoom_1.ply"
pcd_target = o3d.io.read_point_cloud(TLS_column_inliers_merged)
pcd_source = o3d.io.read_point_cloud(BIM_column_inliers_merged)

# Vizualize the column inliers of both dataset
o3d.visualization.draw_geometries([pcd_target])
o3d.visualization.draw_geometries([pcd_source])

##################################### E N D  inlier columns #############################################