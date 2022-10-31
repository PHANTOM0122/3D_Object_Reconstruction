import open3d as o3d
import copy
import cv2
import numpy as np

def draw_registration_result(source, target, transformation, mode=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if mode == None:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_feature(source, feature):
    source_temp = copy.deepcopy(source)
    feature_temp = copy.deepcopy(feature)

    feature_temp.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([source_temp, feature_temp])

# 2개의 좌우 이미지에서 검출된 특징점들을 선들로 이어주어 시각화 하는 함수입니다
def drawlines(img1, img2, lines, pts1, pts2):
    '''
    img1 - image on which we draw the epilines for the points in
    img2 lines - corresponding epilines
    '''

    r, c = img1.shape

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

