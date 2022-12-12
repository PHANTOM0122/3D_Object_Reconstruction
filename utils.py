import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation
from SIFT import *

def transform_3d_points(points_3d, transformation):
    """
    :param pcd:
    :param transformation:
    :return: Transformed pcd
    """
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(points_3d)
    source_pcd.transform(transformation)

    return np.asarray(source_pcd.points)

def geometrically_averaged_points(points, points_unique, points_index):
    """
    Return 3D Geometrically averaged points
    :param points:
    :param points_unique:
    :return:
    """
    avg_points = np.zeros(shape=points_unique.shape)
    for i in range(len(points_unique)):
        temp = []
        # count = 0
        for pt_idx in range(len(points)):
            if points_index[pt_idx] == i:
                temp.append(points[pt_idx])
        temp = np.array(temp)
        avg_points[i] = np.mean(temp, axis=0, dtype=np.float64)
    return avg_points

def preprocess_point_cloud(pcd, voxel_size):

    '''
    :param pcd: Point cloud dataset
    :param voxel_size: Voxel size of dataset
    :return: downsampled point cloud
    '''
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh

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


def R_t_matrix_to_vector(R_t):
    print('R|t shape:', np.array(R_t)[:3][:, :3])
    r = Rotation.from_matrix(np.array(R_t)[:3][:, :3]) # Rodriguess
    r = r.as_quat()

    qx = r[0]
    qy = r[1]
    qz = r[2]
    qw = r[3]

    tx = R_t[0,3]
    ty = R_t[0, 3]
    tz = R_t[0, 3]
    rotation = np.array([qx, qy, qz])
    translation = np.array([tx, ty, tz])
    return rotation, translation

def vector_to_matrix(vector):
    transformation_matrix = np.identity(4)

    rotation_matrix = np.identity(3)
    rvecs = vector[:3]
    cv2.Rodrigues(rvecs, rotation_matrix)

    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0][3] = vector[3]
    transformation_matrix[1][3] = vector[4]
    transformation_matrix[2][3] = vector[5]

    print('transformation_matrix:', transformation_matrix)
    return transformation_matrix

def get_cam_indices(pts1, pts2, pts3):

    cam0 = np.array([0] * len(pts1))
    cam1 = np.array([1] * len(pts2))
    cam2 = np.array([2] * len(pts3))
    # cam3 = np.array([3] * len(pts4))
    result = np.concatenate([cam0, cam1, cam2])
    print('cam indices:', len(result))
    return result

def get_point_indices(pts1_3d, pts2_3d, pts3_3d, pts_unique):
    result = []

    for i in range(len(pts1_3d)):
        for j in range(len(pts_unique)):
            if pts1_3d[i][0] == pts_unique[j][0] and pts1_3d[i][1] == pts_unique[j][1] and pts1_3d[i][2] == pts_unique[j][2]:
                result.append(j)
                break

    for i in range(len(pts2_3d)):
        for j in range(len(pts_unique)):
            if pts2_3d[i][0] == pts_unique[j][0] and pts2_3d[i][1] == pts_unique[j][1] and pts2_3d[i][2] == pts_unique[j][2]:
                result.append(j)
                break

    for i in range(len(pts3_3d)):
        for j in range(len(pts_unique)):
            if pts3_3d[i][0] == pts_unique[j][0] and pts3_3d[i][1] == pts_unique[j][1] and pts3_3d[i][2] == pts_unique[j][2]:
                result.append(j)
                break

    # for i in range(len(pts4_3d)):
    #     for j in range(len(pts_unique)):
    #         if pts4_3d[i][0] == pts_unique[j][0] and pts4_3d[i][1] == pts_unique[j][1] and pts4_3d[i][2] == pts_unique[j][2]:
    #             result.append(j)
    #             break

    print('point indices:', len(result))
    return np.asarray(result)

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine,
                      transformation_icp,
                      information_icp):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def reproject_point2d(points, depth_source_path):
    """
    :param points:
    :return: 3d points
    """
    depth_scaling_factor = 999.99
    focal_length = 597.522 ## mm
    img_center_x = 312.885
    img_center_y = 239.870

    depth = np.array(o3d.io.read_image(depth_source_path), np.float32)
    points_3d = np.array([])
    for point in points:
        u = point[0]
        v = point[1]

        # Normalized image plane -> (u, v, 1) * z = zu, zv, z
        z = np.asarray(depth, dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor # in mm distance
        x = (u - img_center_x) * z / focal_length
        y = (v - img_center_y) * z / focal_length
        points_3d = np.append(points_3d, np.array([x, y, z], dtype=np.float32)).reshape(-1, 3)

    return points_3d


