import os, glob
import numpy as np
import numpy.linalg as LA
from numpy.linalg import inv
import math
import cv2
from enum import Enum
import argparse



def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4). Cam intrinsic
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    z = points[2, :]

    # Normalize x,y coordinate and keep z coordinate.
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        points[2, :] = z

    return points


def read_intrinsic(dir,align_depth=False,verbose=False):
    """
    Args:
        dir (_type_): folder directory
    Returns:
        rgb_dim: [rows,cols]
    """
    K_rgb = np.loadtxt(os.path.join(dir,'intrinsic_color.txt'))
    K_depth = np.loadtxt(os.path.join(dir,'intrinsic_depth.txt'))
    rgb_dim = np.zeros((2),np.int32)    
    depth_dim = np.zeros((2),np.int32)
    
    with open(os.path.join(dir,'sensor_shapes.txt')) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.find('color_height') != -1:
                rgb_dim[0] = int(line.split(':')[-1])
            elif line.find('color_width') != -1:
                rgb_dim[1] = int(line.split(':')[-1])
            elif line.find('depth_height') != -1:
                depth_dim[0] = int(line.split(':')[-1])
            elif line.find('depth_width') != -1:
                depth_dim[1] = int(line.split(':')[-1])
        f.close()
        
    if verbose:
        print('Read color intrinsic: \n',K_rgb)
        print('Read depth intrinsic: \n',K_depth)
        print('Read color shape: \n',rgb_dim)
        print('Read depth shape: \n',depth_dim)
    if align_depth:
        K_rgb = adjust_intrinsic(K_rgb,rgb_dim,depth_dim)
        rgb_dim = depth_dim
    
    return K_rgb,K_depth,rgb_dim,depth_dim

def adjust_intrinsic(intrinsic, raw_image_dim, resized_image_dim):
    '''Adjust camera intrinsics.'''
    import math

    if np.sum(resized_image_dim - raw_image_dim)==0:
        return intrinsic
    resize_width = int(math.floor(resized_image_dim[1] * float(
                    raw_image_dim[0]) / float(raw_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(raw_image_dim[0])
    intrinsic[1, 1] *= float(resized_image_dim[1]) / float(raw_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(resized_image_dim[0] - 1) / float(raw_image_dim[0] - 1)
    intrinsic[1, 2] *= float(resized_image_dim[1] - 1) / float(raw_image_dim[1] - 1)
    return intrinsic    

    
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze() # Nx3
    # print('mean d: {}'.format(uv_depth[:,2].mean()))
    # print(uv_depth.shape)

    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    
    # Project on rgb image
    rgb_uv_depth = view_points(points[:,:3].T,K_rgb,normalize=True).T # Nx3
    # print('mean d: {}'.format(rgb_uv_depth[:,2].mean()))
    count = 0
    
    for i in range(n):
        if rgb_uv_depth[i,0] < 1 or rgb_uv_depth[i,0] > rgb_shape[1]-1 or rgb_uv_depth[i,1] < 0 or rgb_uv_depth[i,1] > rgb_shape[0]-1 or rgb_uv_depth[i,2] < 0.1 or rgb_uv_depth[i,2] > max_depth:
            continue
        align_depth[int(rgb_uv_depth[i,1]),int(rgb_uv_depth[i,0])] = rgb_uv_depth[i,2]
        count +=1
    # print('{}/{} valid in aligned depth'.format(np.count_nonzero(align_depth>0),rgb_shape[0]*rgb_shape[1]))
    
    return align_depth


def project(points, normals,T_wc, K,im_shape,max_depth =3.0,min_depth=1.0,margin=20):
    """
    Args:
        points (np.ndarray): Nx3
        normals (np.ndarray): Nx3
    Output:
        points_uv_all (np.ndarray): Nx3, invalid points are set to -100
        mask (np.ndarray): Nx1, True for valid points
        theta: Nx1, cos theta
    """
    points_uv_all = np.ones((points.shape[0],3),np.float32)-100.0 # Nx3
    normals_uv_all = np.ones((normals.shape[0],3),np.float32)-100.0 # Nx3
    cos_theta_all = np.zeros((points.shape[0]),np.float32) # Nx1
    T_cw = inv(T_wc) # 4x4
    
    # Transform into camera coordinates
    points_homo = np.concatenate([points,np.ones((points.shape[0],1))],axis=1)  # Nx4
    normals_homo = np.concatenate([normals, np.ones((normals.shape[0],1))], axis=1) # 
    points_cam = T_cw.dot(points_homo.T)[:3,:].T  # 3xN, p
    normals_cam_ = T_cw.dot(normals_homo.T)[:3,:].T # 3xN, n
    normals_cam = np.tile(1/ LA.norm(normals_cam_.T,axis=0),(3,1)).T * normals_cam_

    # Cosine(theta) = p \dot n /(|p||n|)   
    # cos_theta = np.sum(points_cam * normals_cam, axis=1)
    # pn_mag = LA.norm(points_cam, axis=1) * LA.norm(normals_cam,axis=1)     
    P_c = points-T_wc[:3,3]
    cos_theta = np.sum(P_c * normals, axis=1)
    pn_mag = LA.norm(P_c, axis=1) * LA.norm(normals, axis=1) 
    cos_theta = cos_theta / pn_mag # [-1,1]
    cos_theta = abs(cos_theta) # [0.0,1.0]
    # cos_theta = np.zeros((points.shape[0]),np.float32)+0.99
    
    # Project into image
    points_uv = view_points(points_cam.T,K,normalize=True).T    # [u,v,d],Nx3
    
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = (points_cam[:,2] > min_depth) &(points_cam[:,2]< max_depth) & (
            points_uv[:,0] > margin) & (points_uv[:,0] < im_shape[1] - margin) & (
            points_uv[:,1] > margin) & (points_uv[:,1] < im_shape[0] - margin)
    # normal_mask = normals_cam[:,2] < 0.0
    # mask = np.logical_and(mask,normal_mask)
    points_uv_all[mask] = points_uv[mask][:,:3]
    normals_uv_all[mask] = normals_cam[mask][:,:3]
    cos_theta_all[mask] = cos_theta[mask]
    # print('mean depth: ',np.mean(points_uv_all[mask][:,2]))

    # print('{}/{} points in camera view'.format(np.count_nonzero(mask),points.shape[0]))
    return mask, points_uv_all, normals_uv_all, cos_theta

def filter_occlusion(points_uv,depth_map,max_dist_gap=0.05, kernel_size=8,
                    min_mask_pixels=0.5, max_variance = 0.2):
    """
    Args:
        points_uv: [N,3]
        depth_map: [H,W,D], float
    Output:
        filter_mask: [N], True for non-occluded points
    """
    # max_variance = 0.01
    
    M = points_uv.shape[0]
    # patch_pixels = (kernel_size*2+1)**2
    # min_mask_pixels = int(patch_pixels*mask_ratio)
    patch_side = kernel_size*2+1
    filter_mask = np.zeros(M,dtype=np.bool_)
    # uv = points_uv[:,:2].astype(np.int32)
    # d_ = depth_map[uv[:,1],uv[:,0]]
    # depth_diff = np.abs(points_uv[:,2] - d_)
    # filter_mask = np.logical_and(depth_diff<max_dist_gap, d_>0.1)
    
    # filter_mask = np.ones(M,dtype=np.bool_)
    
    count_valid = 0
    for i in range(M):
        cx = int(points_uv[i,0])
        cy = int(points_uv[i,1])
        d = points_uv[i,2]
        if d<0.2:continue

        patch = depth_map[cy-kernel_size:cy+kernel_size+1,cx-kernel_size:cx+kernel_size+1]
        mask = patch>0.0
        if np.count_nonzero(mask)/(patch_side*patch_side) <min_mask_pixels:continue
        var = np.var(patch[mask])
        mu = np.mean(patch[mask])
        # mu = depth_map[cy,cx]
        if abs(d-mu)<max_dist_gap and var<max_variance:
            filter_mask[i] = True
        count_valid +=1
                    
    # print('{}/{} non-occluded points'.format(np.count_nonzero(filter_mask),np.count_nonzero(points_uv)))
    return filter_mask


