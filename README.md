# 3D Object Reconstruction with Multi-view RGB-D Images

With RGB-D cameras, we can get multiple RGB and Depth images and convert them to point clouds easily. Leveraging this, we can reconstruct single object with multi-view RGB and Depth images. To acheive this, point clouds from multi-view need to be registered. This task is also known as **point registration"**, whose goal is to find transfomration matrix between source and target point clouds. The alignment consists of two sub-alignment, Initial alignment and Alignment refinement. 

## RGB-D Camera Spec
- Model: Intel realsense D415
- Intrinsic parameters

## Requirements
- Open3D
- Pyrealsense2
- OpenCV
- Numpy

## Depth filtering
Raw depth data are so noisy that depth filtering should be needed. Pyrealsense2 library, developed by Intel, offers filtering methods of depth images. In this project, spatial-filter was used that smoothes noise and preserves edge components in depth images. 

## Initial Alignment
Initial alignment can be acheived through finding transformation matrix between feature points, found by SIFT. The position of 3D points can be estimated with Back-Projection and Depth from depth images. Transformation matrix can be estimated with 3D corresponding feature points from souce and target point clouds, with RANSAC procedure.

## Alignment Refinement
With ICP algorithm implemented in Open3D, refine initial transformation matrix.

## Results <br>
<img src="https://user-images.githubusercontent.com/50229148/198976394-8b62fabf-8240-4684-a482-f698b1f63fdc.gif" width="500" height="300">
