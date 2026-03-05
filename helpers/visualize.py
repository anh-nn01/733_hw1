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


def save_rectified(paths, K, k, out_dir='visuals'):
    """Save undistorted images."""
    save_dir = os.path.join(out_dir,"step4_rectified")
    os.makedirs(save_dir, exist_ok=True)
    dist = np.array([k[0],k[1],0,0,0])

    for p in paths:
        img = cv2.imread(p)
        und = cv2.undistort(img,K,dist)
        name = os.path.basename(p)
        cv2.imwrite(os.path.join(save_dir,f"rect_{name}"),und)


def save_rectified_reprojection(paths, corners, world, K, extr, k, out_dir='visuals'):
    """Rectified image with reprojection overlay."""
    save_dir = os.path.join(out_dir,"step5_rectified_reprojection")
    os.makedirs(save_dir, exist_ok=True)
    dist = np.array([k[0],k[1],0,0,0])

    for i,p in enumerate(paths):
        img = cv2.imread(p)
        rect = cv2.undistort(img,K,dist)

        # remove distortion from detected corners
        und_corners = cv2.undistortPoints(
            corners[i].reshape(-1,1,2).astype(np.float32),
            K,dist,P=K
        ).reshape(-1,2)

        proj = project_points(world,K,*extr[i],[0,0])

        for pt in und_corners:
            cv2.circle(rect,(int(pt[0]),int(pt[1])),6,(0,255,0),2)
        for pt in proj:
            cv2.circle(rect,(int(pt[0]),int(pt[1])),4,(0,0,255),-1)

        name=os.path.basename(p)
        cv2.imwrite(os.path.join(save_dir,f"rect_reproj_{name}"),rect)