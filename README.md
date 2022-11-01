# 3D Object Reconstruction with Multi-view RGB-D Images

With RGB-D cameras, we can get multiple RGB and Depth images and convert them to point clouds easily. Leveraging this, we can reconstruct single object with multi-view RGB and Depth images. To acheive this, point clouds from multi-view need to be registered. This task is also known as **point registration"**, whose goal is to find transfomration matrix between source and target point clouds. The alignment consists of two sub-alignment, Initial alignment and Alignment refinement. 

## RGB-D Camera Spec
- Model: Intel realsense D415
- Intrinsic parameters in 640x480 RGB, Depth image.<br> 
```
K = [[597.522, 0.0, 312.885],
     [0.0, 597.522, 239.870],
     [0.0, 0.0, 1.0]]<br>
```
## Requirements
- Open3D
- Pyrealsense2
- OpenCV
- Numpy

## Align RGB and Depth Image & Depth Filtering
Due to the different position of RGB and Depth lens, aligning them should be done to get exact point clouds. This project used alignment function offered by pyrealsense2 package. Raw depth data are so noisy that depth filtering should be needed. Pyrealsense2 library, developed by Intel, offers filtering methods of depth images. In this project, spatial-filter was used that smoothes noise and preserves edge components in depth images. 

## Pre-Process Point Clouds
Single object might be a part of the scanned data. In order to get points of interested objects, pre-processing should be implemented. Plane-removal, outlier-removal, DBSCAN clustering were executed to extract object. Open3D offers useful functions to filter points. 

## Initial Alignment
Initial alignment can be acheived through finding transformation matrix between feature points, found by SIFT. The position of 3D points can be estimated with Back-Projection and Depth from depth images. Transformation matrix can be estimated with 3D corresponding feature points from souce and target point clouds, with RANSAC procedure.

## Alignment Refinement
With ICP algorithm implemented in Open3D, refine initial transformation matrix.

## Results <br>
The object was reconstructed with 3 different view of RGB-D Images. <br>
<img src="https://github.com/PHANTOM0122/3D_Object_Reconstruction/blob/main/train/align_test21.png" width="320" height="240"/><img src="https://github.com/PHANTOM0122/3D_Object_Reconstruction/blob/main/train/align_test22.png" width="320" height="240"/><img src="https://github.com/PHANTOM0122/3D_Object_Reconstruction/blob/main/train/align_test20.png" width="320" height="240">

The reconstructed point clouds is below. <br>
<img src="https://user-images.githubusercontent.com/50229148/198976394-8b62fabf-8240-4684-a482-f698b1f63fdc.gif" width="500" height="300">
