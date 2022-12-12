import open3d as o3d
import open3d.cpu.pybind.utility
import numpy as np
import cv2
import matplotlib.pyplot as plt
from registration import match_ransac
from utils import get_boundary
########################################################################################################################
# Intrinsic parameter
########################################################################################################################
K = np.array(
     [[597.522, 0.0, 312.885],
     [0.0, 597.522, 239.870],
     [0.0, 0.0, 1.0]], dtype=np.float64)

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.intrinsic_matrix = K
print(intrinsic.intrinsic_matrix)
########################################################################################################################
# Feature matching using SIFT algorithm
########################################################################################################################
# Find transformation matrix from corresponding points based on SIFT
def SIFT_Transformation(img1, img2, depth_img1, depth_img2, source_pcd, target_pcd, distance_ratio=0.6):

    # Read image from path
    imgL = cv2.imread(img1)
    imgR = cv2.imread(img2)
    depthL = np.array(o3d.io.read_image(depth_img1), np.float32)
    depthR = np.array(o3d.io.read_image(depth_img2), np.float32)

    # Clip depth value
    threshold = 3000  # 3m limit
    left_idx = np.where(depthL > threshold)
    right_idx = np.where(depthR > threshold)
    depthL[left_idx] = threshold
    depthR[right_idx] = threshold

    # Intel RealSense D415
    depth_scaling_factor = 999.99
    focal_length = 597.522 ## mm
    img_center_x = 312.885
    img_center_y = 239.870

    # sift = cv2.xfeatures2d.SIFT_create() # OpenCV 4.5 미만 버젼 사용중일 시
    sift = cv2.SIFT_create() # OpenCV 4.5 이상의 버전 사용중일 시

    # Find keypoints and descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    '''
    # FLANN Parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # FLANN Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matching
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []
    kp1_1 = []
    kp2_1 = []
    distance_ratio = 1.2
    # Get Matched points under distance's threshold
    for i, (m, n) in enumerate(matches[:10]):
        if m.distance < distance_ratio * n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            kp1_1.append(kp1[m.queryIdx])
            kp2_1.append(kp2[m.trainIdx])
    '''
    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    good_matches = []
    pts1 = []
    pts2 = []
    kp1_1 = []
    kp2_1 = []
    source_x_min, source_x_max, source_y_min, source_y_max = get_boundary(source_pcd)
    target_x_min, target_x_max, target_y_min, target_y_max = get_boundary(target_pcd)
    print(source_x_min, source_x_max, source_y_min, source_y_max)
    print(target_x_min, target_x_max, target_y_min, target_y_max)

    # depth map에서 위치의 min, max x, y 찾아서 마스킹해서 outlier 제거
    for i, (m, n) in enumerate(matches):
        if m.distance < distance_ratio * n.distance: # 0.6 for castard,
            if (kp1[m.queryIdx].pt[0] >= source_x_min and kp1[m.queryIdx].pt[0] <= source_x_max):
                if (kp1[m.queryIdx].pt[1] >= source_y_min and kp1[m.queryIdx].pt[1] <= source_y_max):
                    if (kp2[m.trainIdx].pt[0] >= target_x_min and kp2[m.trainIdx].pt[0] <= target_x_max):
                        if (kp2[m.trainIdx].pt[1] >= target_y_min and kp2[m.trainIdx].pt[1] <= target_y_max):
                            good_matches.append([m])
                            pts1.append(kp1[m.queryIdx].pt) # Source pcd
                            pts2.append(kp2[m.trainIdx].pt) # Target pcd
                            kp1_1.append(kp1[m.queryIdx])
                            kp2_1.append(kp2[m.trainIdx])
                            matchesMask[i] = [1, 0]

    # Print number of matched feature points
    print('Matched Num:', len(matches))
    print('Good Matched Num:', len(good_matches))
    print('Left Keypoint num:', len(kp1_1))
    print('Right Keypoint num:', len(kp2_1))

    img_matched = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, good_matches, None, matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0), flags=2)
    cv2.imshow('img_matched', img_matched)
    cv2.waitKey(0)

    # Set array for keypoints
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # Correspondence set
    matches_index = np.array([])
    for i in range(len(pts1)):
        matches_index = np.append(matches_index, np.array([i, i]))
    matches_index = matches_index.reshape(-1, 2)
    correspondence_points = open3d.utility.Vector2iVector(matches_index)

    pts1_3d = []
    pts2_3d = []

    for i in range(pts1.shape[0]):
        # Image plane -> 픽셀값
        u = np.float64(pts1[i][0])
        v = np.float64(pts1[i][1])

        # Normalized image plane -> (u, v, 1) * z = zu, zv, z
        z = np.asarray(depthL, dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor # in mm distance
        x = (u - img_center_x) * z / focal_length
        y = (v - img_center_y) * z / focal_length
        pts1_3d = np.append(pts1_3d, np.array([x, y, z], dtype=np.float32))

    for i in range(pts2.shape[0]):
        # Image plane
        u = np.float64(pts2[i][0])
        v = np.float64(pts2[i][1])

        # Normalized image plane
        z = np.asarray(depthR, dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor # in mm distance
        x = (u - img_center_x) * z / focal_length
        y = (v - img_center_y) * z / focal_length
        pts2_3d = np.append(pts2_3d, np.array([x, y, z], dtype=np.float32))

    pts1_3d = pts1_3d.reshape(-1, 3)
    pts2_3d = pts2_3d.reshape(-1, 3)
    print(pts1_3d.shape, pts2_3d.shape)

    # Declare point cloud
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    #  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
    #  pc_color: array(Nx3), each row composed with R G,B in the rage of 0 ~ 1
    pc_points1 = np.array(pts1_3d, np.float32)
    pc_points2 = np.array(pts2_3d, np.float32)
    pc_color1 = np.array([], np.float32)
    pc_color2 = np.array([], np.float32)

    for i in range(pts1.shape[0]):
        u = np.int32(pts1[i][0])
        v = np.int32(pts1[i][1])
        # pc_colors
        pc_color1 = np.append(pc_color1, np.array(np.float32(imgL[v][u] / 255)))
        pc_color1 = np.reshape(pc_color1, (-1, 3))

    for i in range(pts2.shape[0]):
        u = np.int32(pts2[i][0])
        v = np.int32(pts2[i][1])
        # pc_colors
        pc_color2 = np.append(pc_color2, np.array(np.float32(imgR[v][u] / 255)))
        pc_color2 = np.reshape(pc_color2, (-1, 3))

    # add position and color to point cloud
    pcd1.points = o3d.utility.Vector3dVector(pc_points1)
    pcd1.colors = o3d.utility.Vector3dVector(pc_color1)
    pcd2.points = o3d.utility.Vector3dVector(pc_points2)
    pcd2.colors = o3d.utility.Vector3dVector(pc_color2)
    '''
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    R_t = p2p.compute_transformation(
        pcd1,
        pcd2,
        correspondence_points
    )
    '''
    R_t = match_ransac(pts1_3d, pts2_3d, tol=0.1)


    print("Transformation is:")
    print(R_t)

    return R_t, pcd1, pcd2, pts1, pts2, pts1_3d, pts2_3d
