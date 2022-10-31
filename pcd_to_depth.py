import open3d as o3d

# Project into images
pcd_list = ["./pcd_o3d/castard_20.pcd", "./pcd_o3d/castard_21.pcd", "./pcd_o3d/castard_22.pcd"]
jpg_list = ["file20.jpg", "file21.jpg", "file22.jpg"]
depth_list = ["file20_depth.png", "file21_depth.png", "file22_depth.png"]

for i in range(3):
   pcd_temp = o3d.io.read_point_cloud(pcd_list[i], format="pcd")
   pcd_temp.transform([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
   vis = o3d.visualization.Visualizer()
   vis.create_window()
   vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
   vis.get_render_option().point_size = 3.0
   vis.add_geometry(pcd_temp)

   # Capture rgb image
   vis.capture_screen_image(jpg_list[i], do_render=True)
   # Capture depth image
   vis.capture_depth_image(depth_list[i], do_render=True)
   vis.destroy_window()

