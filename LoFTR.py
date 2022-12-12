import numpy as np
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from kornia_moons.feature import draw_LAF_matches
from registration import match_ransac
from utils import get_boundary

# Intel RealSense D415
depth_scaling_factor = 999.99
focal_length = 597.522  ## mm
img_center_x = 312.885
img_center_y = 239.870

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255. # Normalized Tensor
    img = K.color.bgr_to_rgb(img)
    return img

def LoFTR_Transformation(img1_path, img2_path, depth_img1_path, depth_img2_path, pcd1, pcd2):

    img1 = load_torch_image(img1_path)
    img2 = load_torch_image(img2_path)
    imgL = cv2.imread(img1_path)
    imgR = cv2.imread(img2_path)
    depthL = np.array(o3d.io.read_image(depth_img1_path), np.float32)
    depthR = np.array(o3d.io.read_image(depth_img2_path), np.float32)

    # Define matcher
    matcher = KF.LoFTR(pretrained='outdoor') # indoor or outdoor

    # LofTR works on grayscale images only
    input_dict = {"image0": K.color.rgb_to_grayscale(img1),
                  "image1": K.color.rgb_to_grayscale(img2)}

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    for k,v in correspondences.items():
        print (k)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    print('mkpts0 shape:', mkpts0.shape)
    print('mkpts1 shape', mkpts1.shape)

    pts1 = []
    pts2 = []
    source_x_min, source_x_max, source_y_min, source_y_max = get_boundary(pcd1)
    target_x_min, target_x_max, target_y_min, target_y_max = get_boundary(pcd2)

    # depth map에서 위치의 min, max x, y 찾아서 마스킹해서 outlier 제거
    for idx in range(len(mkpts0)):
        if (mkpts0[idx][0] >= source_x_min and mkpts0[idx][0] <= source_x_max):
            if (mkpts0[idx][1] >= source_y_min and mkpts0[idx][1] <= source_y_max):
                if (mkpts1[idx][0] >= target_x_min and mkpts1[idx][0] <= target_x_max):
                    if (mkpts1[idx][1] >= target_y_min and mkpts1[idx][1] <= target_y_max):
                        pts1.append(mkpts0[idx])
                        pts2.append(mkpts1[idx])
    print('constrained point:', len(pts1))

    # Select only inlier
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    mkpts0 = pts1
    mkpts1 = pts2
    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    # Visualize matching result
    '''
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),

        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None,
                   'feature_color': (0.2, 0.5, 1), 'vertical': False})
    plt.show()
    '''
    # Correspondence set
    matches_index = np.array([])
    for i in range(len(pts1)):
        matches_index = np.append(matches_index, np.array([i, i]))
    matches_index = matches_index.reshape(-1, 2)
    correspondence_points = o3d.utility.Vector2iVector(matches_index)

    # 3D points
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

    R_t = match_ransac(pts1_3d, pts2_3d, tol=0.1)

    print("Transformation is:")
    print(R_t)

    return R_t, pcd1, pcd2