import open3d as o3d
import numpy as np
import cv2

# Intel RealSense D415
depth_scaling_factor = 1000
focal_length = 597.522  ## mm
img_center_x = 312.885
img_center_y = 239.870

img = cv2.imread('./train/align_test7.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
depth = o3d.io.read_image('./train/align_test_depth7.png')
depth = np.asarray(depth, np.float32)
threshold = 3000  # 1m limit
idx = np.where(depth > threshold)
depth[idx] = 0
print(depth.max(), depth.min())

original_pcd = o3d.geometry.PointCloud()
original_pcd_pos = []
original_pcd_color = []

for v in range(img.shape[0]): # height
     for u in range(img.shape[1]): # width
         # Normalized image plane -> (u, v, 1) * z = zu, zv, z
         z = depth[v][u] / depth_scaling_factor # mm
         x = (u - img_center_x) * z / focal_length
         y = (v - img_center_y) * z / focal_length

         original_pcd_pos.append([x, y, z])
         original_pcd_color.append(img[v][u] / 255)
print(x, y, z)
original_pcd_pos = np.array(original_pcd_pos, dtype=np.float32)
original_pcd_color = np.array(original_pcd_color, dtype=np.float32)

original_pcd.points = o3d.utility.Vector3dVector(original_pcd_pos)
original_pcd.colors = o3d.utility.Vector3dVector(original_pcd_color)

# Save point cloud
o3d.io.write_point_cloud('./pcd_np/pcd7.pcd', original_pcd)
o3d.visualization.draw_geometries([original_pcd])
