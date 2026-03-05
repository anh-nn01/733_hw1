"""
Author: Anh N. Nhu
UID: 119 385 173

Pipeline:
	1. Detect checkerboard corners
	2. 	a) Estimate homography for each image
		b) Compute intrinsic matrix (closed-form solution)
		c) Compute camera poses
	3. Optimization using error minimization
	4. Undistort images
	5. Compute reprojection error
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
from scipy.optimize import least_squares

import matplotlib

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(f'{filepath}/helpers/'))
from helpers.core import *
from helpers.visualize import *


#############################
# Checkerboard parameters   #
#############################
BOARD_ROWS = 6        
BOARD_COLS = 9     
SQUARE_SIZE  = 21.5  # given

##################
# CMD arguments	 #
##################
parser = argparse.ArgumentParser(description="Zhang's Camera Calibration")
parser.add_argument('--img_dir', type=str, default='Calibration_Imgs',
					help='Directory containing calibration images')
# parser.add_argument('--out_dir', type=str, default='results',
# 					help='Output directory for results')
filepath = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(filepath, 'visuals/')
args = parser.parse_args()
os.makedirs(out_dir, exist_ok=True)
matplotlib.use('Agg')


if __name__ == "__main__":
	paths = sorted(
		glob.glob(os.path.join(args.img_dir,"*.jpg")) +
		glob.glob(os.path.join(args.img_dir,"*.png"))
	)
	if not paths:
		raise Exception('No Caliberation Image found!!!')

	print("Step 0: Image loading:")
	print(f"\tTotal: {len(paths)} images")
	print('*'*50)
	print()

	print('Step 1: Corneer detection')
	corners, shapes, valid = detect_corners(
		paths,
		(BOARD_COLS,BOARD_ROWS)
	)
	if len(corners) < 3:
		raise Exception("Need >= 3 valid views.")
	# visualize
	save_corners(valid, corners, (BOARD_COLS,BOARD_ROWS), out_dir)
	print('*'*50)
	print()



	print('Step 2: Compute Homography and Solve for Initial Intrinsics')
	world = create_world_points(
		(BOARD_COLS,BOARD_ROWS),
		SQUARE_SIZE
	)
	Hs = compute_homographies(world,corners)
	K = estimate_intrinsics(Hs)
	M = estimate_extrinsics(K,Hs)
	k = np.array([0.0,0.0])
	print("\nInitial reprojection error:")
	err0 = reprojection_error(
		world, corners, K, M, k)
	print(np.mean(err0))
	# visualization
	save_reprojection(valid, corners, world, K, M, k, "step2_initial_reprojection", out_dir)
	print('*'*50)
	print()

	

	print('Step 3: Optimization')
	K_opt, extr_opt, k_opt = run_optimization(
		K, M, k, world, corners
	)
	with np.printoptions(precision=2, suppress=True):
		print("\nOptimized intrinsics:")
		print(K_opt)
		print("\nDistortion:")
		print(k_opt)
		print('*'*50)
		print()
	# visualization
	save_reprojection(valid, corners, world, K_opt, extr_opt, k_opt, "step3_optimized_reprojection", out_dir)



	print('Step 4+5: Image Undistortion + Final evals')
	err = reprojection_error(world,corners,K_opt,extr_opt,k_opt)
	print("Final reprojection error:")
	print(np.mean(err))
	# visualization
	scale = 5.0 # 1.5
	h_board, w_board = int((BOARD_COLS) * SQUARE_SIZE * scale), int((BOARD_ROWS) * SQUARE_SIZE * scale)
	save_rectified(valid, Hs, K_opt, k_opt, h_board, w_board, scale, out_dir)
	save_rectified_reprojection(
		valid, corners, world, K_opt, extr_opt, k_opt, 
		Hs, h_board, w_board, scale, out_dir
	)