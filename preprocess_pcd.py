import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def rgbd_to_pcd(count):

    source_color = o3d.io.read_image('./train/spyderman2/rgb/align_test%d.png'%count)
    source_depth = o3d.io.read_image('./train/spyderman2/depth/align_test_depth%d.png'%count)

    K = np.array(
         [[597.522, 0.0, 312.885],
         [0.0, 597.522, 239.870],
         [0.0, 0.0, 1.0]], dtype=np.float64)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = K

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth, depth_scale=1000, convert_rgb_to_intensity=False, depth_trunc=1)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)
    # o3d.io.write_point_cloud('./pcd_o3d/spyderman2/spyderman2_%d.pcd' % count, pcd)

    # Plane Segmentation
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    # o3d.visualization.draw_geometries([outlier_cloud])

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            outlier_cloud.cluster_dbscan(eps=0.02, # Epsilon defines the distance between to neighbors in a cluster
                               min_points=500, # minimum number of points required to form a cluster
                               print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # clusters = labels
    indexes = np.where(labels == 0)

    # Extract Interest point clouds
    interest_pcd = o3d.geometry.PointCloud()
    interest_pcd.points = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.points, np.float32)[indexes])
    interest_pcd.colors = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.colors, np.float32)[indexes])

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([interest_pcd])

    # Plane Segmentation for floor
    # plane_model, inliers = interest_pcd.segment_plane(distance_threshold=0.001,
    #                                          ransac_n=3,
    #                                          num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # inlier_cloud = interest_pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = interest_pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    y = np.asarray(outlier_cloud.points)[:, 1]
    y_mean = np.mean(y)
    # plt.plot(y)
    # plt.show()
    # idx = np.array([i for i in range(len(z))], dtype=np.int)
    idx = np.where(y < 0.18)[0]
    idx = np.asarray(idx, dtype=np.int)

    interest_pcd = outlier_cloud.select_by_index(list(idx))
    # o3d.visualization.draw_geometries([interest_pcd])

    print("Statistical oulier removal")
    cl, ind = interest_pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.8)
    # o3d.visualization.draw_geometries([cl])

    print("Radius oulier removal")
    cl, ind = cl.remove_radius_outlier(nb_points=100, radius=0.01)
    o3d.visualization.draw_geometries([cl])

    o3d.io.write_point_cloud('./pcd_o3d/spyderman2/spyderman2%d.pcd'%count, cl)

if __name__ == '__main__':
    for i in range(1, 33):
        rgbd_to_pcd(i)