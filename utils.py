import numpy as np
import open3d as o3d

def get_boundary(source_pcd):

    # Intrinsic parameter for Realsense D415
    depth_scaling_factor = 999.99
    focal_length = 597.522
    img_center_x = 312.885
    img_center_y = 239.870

    x_min = np.min(np.asarray(source_pcd.points)[:, 0])
    x_max = np.max(np.asarray(source_pcd.points)[:, 0])
    y_min = np.min(np.asarray(source_pcd.points)[:, 1])
    y_max = np.max(np.asarray(source_pcd.points)[:, 1])

    x_min_idx = np.where(np.asarray(source_pcd.points)[:, 0] == x_min)
    x_max_idx = np.where(np.asarray(source_pcd.points)[:, 0] == x_max)
    y_min_idx = np.where(np.asarray(source_pcd.points)[:, 1] == y_min)
    y_max_idx = np.where(np.asarray(source_pcd.points)[:, 1] == y_max)

    u_min = x_min * focal_length / (np.asarray(source_pcd.points)[x_min_idx][0][2]) + img_center_x
    u_max = x_max * focal_length / (np.asarray(source_pcd.points)[x_max_idx][0][2]) + img_center_x
    v_min = y_min * focal_length / (np.asarray(source_pcd.points)[y_min_idx][0][2]) + img_center_y
    v_max = y_max * focal_length / (np.asarray(source_pcd.points)[y_max_idx][0][2]) + img_center_y

    return u_min, u_max, v_min, v_max