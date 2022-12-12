import numpy as np
import open3d as o3d
from SIFT import SIFT_Transformation # SIFT feature points based registration
from ORB import ORB_Transformation # ORB feature points based registration
from LoFTR import LoFTR_Transformation # LoFTR method based registration
import matplotlib.pyplot as plt

# Load point clouds
def load_point_clouds(voxel_size=0.0, pcds_paths=None):
    pcds = []
    print('len demo_icp_pcds_paths:', len(pcds_paths))
    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

def load_orginal_point_clouds(voxel_size=0.0, pcds_paths=None):
    pcds = []
    print('len demo_icp_pcds_paths:', len(pcds_paths))
    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcds.append(pcd)
    return pcds

def relative_camera_poses_all(rgb_lists, depth_lists, pcd_lists):

    pose_list = []
    num_multiview = len(rgb_lists)

    for i in range(num_multiview-1):
        j = i + 1
        transformation, pcd1_features, source_pcd1_features, pts1, pts_source_1, pts1_3d, pts_source1_3d = SIFT_Transformation(
            rgb_lists[i], rgb_lists[j],
            depth_lists[i], depth_lists[j],
            pcd_lists[i], pcd_lists[j],
            distance_ratio=0.7)
        pose_list.append(transformation)

    return np.asarray(pose_list)

def relative_camera_poses_select(start_idx, end_idx, pose_list):

    result = np.identity(4)
    for idx in range(start_idx, end_idx):
        result = pose_list[idx] @ result
    return result

def pairwise_registration(source, target, init_trans):
    print("Apply point-to-plane ICP")
    source.estimate_normals()
    target.estimate_normals()
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine, relative_camera_poses=None):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds) # 16
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print('source id:', source_id)
            print('target id:', target_id)
            threshold = 0.001

            init_trans = np.identity(4)
            transformation_icp, information_icp = pairwise_registration(pcds_down[source_id], pcds_down[target_id],
                                                                        init_trans)

            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case

                init_trans, pcd1_features, source_pcd1_features, pts1, pts_source_1, pts1_3d, pts_source1_3d = SIFT_Transformation(
                    rgb_path[source_id], rgb_path[target_id],
                    depth_path[source_id], depth_path[target_id],
                    origin_pcds[source_id], origin_pcds[target_id],
                    distance_ratio=0.9)
                # init_trans = relative_camera_poses_select(start_idx=source_id, end_idx=target_id, pose_list=relative_camera_poses)

                # origin_pcds[source_id].estimate_normals()
                # origin_pcds[target_id].estimate_normals()

                icp_fine = o3d.pipelines.registration.registration_icp(
                    origin_pcds[source_id], origin_pcds[target_id], threshold,
                    init_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                transformation_icp = icp_fine.transformation

                information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    origin_pcds[source_id], origin_pcds[target_id], threshold,
                    transformation_icp)

                # visualize transformation icp result
                # draw_registration_result(pcds_down[source_id], pcds_down[target_id], transformation_icp, mode='rgb')

                odometry = np.dot((transformation_icp), odometry)

                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # Connect any non-neighboring nodes

                # transformation_icp = relative_camera_poses_select(start_idx=source_id, end_idx=target_id, pose_list=relative_camera_poses)
                # information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                #     origin_pcds[source_id], origin_pcds[target_id], threshold,
                #     transformation_icp)

                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    # Loop closure
    source_id = n_pcds - 1
    target_id = 0
    init_trans, pcd1_features, source_pcd1_features, pts1, pts_source_1, pts1_3d, pts_source1_3d = SIFT_Transformation(
        rgb_path[source_id], rgb_path[target_id],
        depth_path[source_id], depth_path[target_id],
        origin_pcds[source_id], origin_pcds[target_id],
        distance_ratio=0.9)

    origin_pcds[source_id].estimate_normals()
    origin_pcds[target_id].estimate_normals()

    threshold = 0.01
    icp_fine = o3d.pipelines.registration.registration_icp(
        origin_pcds[source_id], origin_pcds[target_id], threshold,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        origin_pcds[source_id], origin_pcds[target_id], threshold,
        transformation_icp)

    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(n_pcds-1,
                                                 0,
                                                 transformation_icp,
                                                 information_icp,
                                                 uncertain=True))
    return pose_graph

if __name__ == "__main__":

    object = "spyderman" # "castard" or "spyderman"

    # Path
    if object == "castard":
        depth_path = ['./train/castard/depth/align_test_depth%d.png' % i for i in range(1, 6)]
        rgb_path = ['./train/castard/rgb/align_test%d.png' % i for i in range(1, 6)]
        pcds_paths = ['./pcd_o3d/castard/box1.pcd', './pcd_o3d/castard/box2.pcd', './pcd_o3d/castard/box3.pcd',
                      './pcd_o3d/castard/box4.pcd', './pcd_o3d/castard/box5.pcd', './pcd_o3d/castard/box6.pcd',
                      './pcd_o3d/castard/box7.pcd', './pcd_o3d/castard/box8.pcd', './pcd_o3d/castard/box9.pcd',
                      './pcd_o3d/castard/box10.pcd', './pcd_o3d/castard/box11.pcd', './pcd_o3d/castard/box12.pcd',
                      './pcd_o3d/castard/box13.pcd', './pcd_o3d/castard/box14.pcd', './pcd_o3d/castard/box15.pcd',
                      './pcd_o3d/castard/box16.pcd', './pcd_o3d/castard/box17.pcd', './pcd_o3d/castard/box18.pcd',
                      './pcd_o3d/castard/box19.pcd', './pcd_o3d/castard/box20.pcd']
    elif object == "spyderman":
        depth_path = ['./train/spyderman/depth/align_test_depth%d.png' % i for i in range(1, 17)]
        rgb_path = ['./train/spyderman/rgb/align_test%d.png' % i for i in range(1, 17)]
        pcds_paths = ['./pcd_o3d/spyderman/spyderman1.pcd', './pcd_o3d/spyderman/spyderman2.pcd',
                      './pcd_o3d/spyderman/spyderman3.pcd', './pcd_o3d/spyderman/spyderman4.pcd',
                      './pcd_o3d/spyderman/spyderman5.pcd', './pcd_o3d/spyderman/spyderman6.pcd',
                      './pcd_o3d/spyderman/spyderman7.pcd', './pcd_o3d/spyderman/spyderman8.pcd',
                      './pcd_o3d/spyderman/spyderman9.pcd', './pcd_o3d/spyderman/spyderman10.pcd',
                      './pcd_o3d/spyderman/spyderman11.pcd', './pcd_o3d/spyderman/spyderman12.pcd',
                      './pcd_o3d/spyderman/spyderman13.pcd', './pcd_o3d/spyderman/spyderman14.pcd',
                      './pcd_o3d/spyderman/spyderman15.pcd', './pcd_o3d/spyderman/spyderman16.pcd'
                      ]

    # Define voxel size to Downsample
    voxel_size = 0.001
    origin_pcds = load_orginal_point_clouds(voxel_size, pcds_paths)
    pcds_down = load_point_clouds(voxel_size, pcds_paths)
    o3d.visualization.draw_geometries(pcds_down)

    relative_camera_poses = relative_camera_poses_all(rgb_path, depth_path, origin_pcds)
    print('pose shape:', relative_camera_poses.shape)

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 10
    max_correspondence_distance_fine = voxel_size * 1
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine,
                                       )

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    print("Transform points and display")
    accumulated_pcd = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        accumulated_pcd += pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries([accumulated_pcd])
    # o3d.io.write_point_cloud('accumulated_%s.pcd'%object, accumulated_pcd)

    y = np.asarray(accumulated_pcd.points)[:, 1]
    y_mean = np.mean(y)
    plt.plot(y)
    plt.show()
    # idx = np.array([i for i in range(len(z))], dtype=np.int)
    idx = np.where(y < 0.138)[0]
    idx = np.asarray(idx, dtype=np.int)
    interest_pcd = accumulated_pcd.select_by_index(list(idx))
    o3d.visualization.draw_geometries([interest_pcd])

    # Render
    vis = o3d.visualization.Visualizer()
    vis.create_window('3DReconstructed')

    for p in pcds_down:
        vis.add_geometry(p)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 1.5

    vis.run()
    vis.destroy_window()