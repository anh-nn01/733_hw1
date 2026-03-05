"""
Visualization helpers

Author: Anh N. Nhu
Mar. 03, 2026
"""
import os
import sys
import cv2
import numpy as np
import matplotlib

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(f'{filepath}/../helpers/'))
from core import *

#############################
# Visualization Utilities   #
#############################
def save_corners(paths, corners, pattern_size, out_dir='visuals'):
    """Save checkerboard detections."""
    save_dir = os.path.join(out_dir, "step1_corners")
    os.makedirs(save_dir, exist_ok=True)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        pts = corners[i].reshape(-1,1,2).astype(np.float32)
        cv2.drawChessboardCorners(img, pattern_size, pts, True)
        name = os.path.basename(p)
        cv2.imwrite(os.path.join(save_dir, f"corners_{name}"), img)


def save_reprojection(paths, corners, world, K, extr, k, folder_name, out_dir='visuals'):
    """Save images with detected and reprojected corners."""
    save_dir = os.path.join(out_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        proj = project_points(world, K, *extr[i], k)

        # detected (green)
        for pt in corners[i]:
            cv2.circle(img,(int(pt[0]),int(pt[1])),6,(0,255,0),2)
        # projected (red)
        for pt in proj:
            cv2.circle(img,(int(pt[0]),int(pt[1])),4,(0,0,255),-1)

        name = os.path.basename(p)
        cv2.imwrite(os.path.join(save_dir,f"reproj_{name}"),img)


def save_rectified(paths, list_homography, K, k, h_board, w_board, out_dir='visuals'):
    """Save undistorted images."""
    save_dir = os.path.join(out_dir,"step4_rectified")
    os.makedirs(save_dir, exist_ok=True)
    dist = np.array([k[0],k[1],0,0,0])

    for idx, p in enumerate(paths):
        img = cv2.imread(p)
        img_undistorted = cv2.undistort(img,K,dist)
        rect = cv2.warpPerspective(
            img_undistorted, np.linalg.inv(list_homography[idx]), 
            (h_board, w_board)
        ) # fix: too much blank space
        # print(rect.shape)
        name = os.path.basename(p)
        # cv2.imwrite(os.path.join(save_dir,f"rect_{name}"),img_undistorted)
        cv2.imwrite(os.path.join(save_dir,f"rect_{name}"),rect)


def save_rectified_reprojection(
    paths, corners, world, K, extr, k, 
    list_homography, h_board, w_board, 
    out_dir='visuals'
):
    """Rectified image with reprojection overlay."""
    save_dir = os.path.join(out_dir, "step5_rectified_reprojection")
    os.makedirs(save_dir, exist_ok=True)
    dist = np.array([k[0], k[1], 0, 0, 0], dtype=np.float32)

    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
            
        # 1. Address radial distortion 
        und = cv2.undistort(img, K, dist)
        # 2.Image rectification with homography
        H_inv = np.linalg.inv(list_homography[i])
        rectified_img = cv2.warpPerspective(und, H_inv, (h_board, w_board))
        # 3. Transform DETECTED corners into the rectified space
        # We take the undistorted image points and apply H_inv to see them on the 'flat' board
        und_pts = cv2.undistortPoints(
            corners[i].reshape(-1, 1, 2).astype(np.float32),
            K, dist, P=K
        )
        # Apply perspective transform to the points: (Image -> World/Rectified)
        rect_corners = cv2.perspectiveTransform(und_pts, H_inv).reshape(-1, 2)
        # 4. Transform MATHEMATICAL projections into the rectified space
        # Since 'project_points' returns pixel coordinates, we apply H_inv to those too
        proj_pts = project_points(world, K, *extr[i], k).reshape(-1, 1, 2).astype(np.float32)
        rect_proj = cv2.perspectiveTransform(proj_pts, H_inv).reshape(-1, 2)
        # 5. Draw the results on the rectified image
        # Detected (undistorted then rectified) in Green
        for pt in rect_corners:
            cv2.circle(rectified_img, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), 2)
        # Projected (calculated then rectified) in Red
        for pt in rect_proj:
            cv2.circle(rectified_img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
        name = os.path.basename(p)
        cv2.imwrite(os.path.join(save_dir, f"rect_reproj_{name}"), rectified_img)