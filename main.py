import open3d as o3d
import numpy as np

from SIFT import SIFT_Transformation
from SURF import SURF_Transformation
from ORB import ORB_Transformation
from plot import draw_registration_result, draw_feature
from utils import get_boundary
from registration import *

if __name__ == '__main__':

    # Set image path
    source_img_path = './train/align_test21.png'
    depth_source_path = './train/align_test_depth21.png'
    img1_path = './train/align_test20.png'
    depth_img1_path = './train/align_test_depth20.png'
    img2_path = './train/align_test21.png'
    depth_img2_path = './train/align_test_depth21.png'
    img3_path = './train/align_test22.png'
    depth_img3_path = './train/align_test_depth22.png'

    # Set feature matching method
    feature_matching = 'sift' # sift, surf, orb, fpfh

    # Read point cloud
    source_pcd = o3d.io.read_point_cloud('./pcd_o3d/castard_21.pcd')
    pcd1 = o3d.io.read_point_cloud('./pcd_o3d/castard_20.pcd')
    pcd2 = o3d.io.read_point_cloud('./pcd_o3d/castard_21.pcd')
    pcd3 = o3d.io.read_point_cloud('./pcd_o3d/castard_22.pcd')

    # Feature points
    pcd1_features = o3d.geometry.PointCloud()
    pcd2_features_1 = o3d.geometry.PointCloud()
    pcd2_features_2 = o3d.geometry.PointCloud()
    pcd3_features = o3d.geometry.PointCloud()

    # Get transformation matrix from Feature matching
    if feature_matching == 'sift':
        init_trans1, pcd1_features, pcd2_features_1 = SIFT_Transformation(img1_path, img2_path, depth_img1_path, depth_img2_path, pcd1, pcd2)
        init_trans2, pcd3_features, pcd2_features_2 = SIFT_Transformation(img3_path, img2_path, depth_img3_path, depth_img2_path, pcd3, pcd2)
    elif feature_matching == 'surf':
        init_trans1 = SURF_Transformation(img1_path, img2_path, depth_img1_path, depth_img2_path)
        init_trans2 = SURF_Transformation(img3_path, img2_path, depth_img3_path, depth_img2_path)
    elif feature_matching == 'orb':
        init_trans1, pcd1_features, pcd2_features_1 = ORB_Transformation(img1_path, img2_path, depth_img1_path, depth_img2_path)
        init_trans2, pcd3_features, pcd2_features_2 = ORB_Transformation(img3_path, img2_path, depth_img3_path, depth_img2_path)
    elif feature_matching == 'fpfh':
        # Fast global registration with FPFH features
        voxel_size = 0.01 # 1cm for this dataset

        # Extract fpfh features
        pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd1, voxel_size)
        pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd2, voxel_size)
        pcd3_down, pcd3_fpfh = preprocess_point_cloud(pcd3, voxel_size)

        init_trans1 = execute_global_registration(pcd1_down, pcd2_down, pcd1_fpfh, pcd2_fpfh, voxel_size)
        init_trans2 = execute_global_registration(pcd3_down, pcd2_down, pcd3_fpfh, pcd2_fpfh, voxel_size)

    '''
    # Colored icp registration
    voxel_size = [4, 2, 1]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_size[scale]

        source_down = pcd1.voxel_down_sample(radius)
        target_down = pcd2.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

        registration1 = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, init_trans1,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = registration1.transformation
    '''

    # ICP refinement of transformation matrix
    print("After ICP refinement, transformation is:")
    threshold = 0.01  # 1 -> 1m distance threshold
    registration1 = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold, init_trans1,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    registration2 = o3d.pipelines.registration.registration_icp(
           pcd3, pcd2, threshold, init_trans2,
           o3d.pipelines.registration.TransformationEstimationPointToPoint(),
           o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    print(registration1.transformation)
    print(registration2.transformation)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualize registration's result
    pcd1_features.paint_uniform_color([1.0, 0, 0])
    pcd2_features_1.paint_uniform_color([0, 0, 1.0])
    o3d.visualization.draw_geometries([pcd1_features, pcd2_features_1, pcd1, pcd2])
    draw_registration_result(pcd1, pcd2, init_trans1, mode='color')
    draw_registration_result(pcd1, pcd2, registration1.transformation, mode='color')

    pcd3_features.paint_uniform_color([1.0, 0, 0])
    pcd2_features_2.paint_uniform_color([0, 0, 1.0])
    o3d.visualization.draw_geometries([pcd2_features_2, pcd3_features, pcd2, pcd3])
    draw_registration_result(pcd3, pcd2, init_trans2, mode='color')
    draw_registration_result(pcd3, pcd2, registration2.transformation, mode='color')

    # Save accumulated pcd
    pcd2 += pcd1.transform(init_trans1)
    pcd2 += pcd3.transform(registration2.transformation)
    o3d.io.write_point_cloud('accumulated_pcd.pcd', pcd2)
    o3d.visualization.draw_geometries([pcd2])