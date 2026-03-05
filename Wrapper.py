"""
Camera Calibration using Zhang's Method

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
out_dir = './visuals/'
args = parser.parse_args()
os.makedirs(out_dir, exist_ok=True)
matplotlib.use('Agg')


#############################
# Corner Detection			#
#############################
def detect_corners(image_paths, pattern_size):
	"""
	Detect checkerboard corners in all images.

	Output:
		corners_list : list of Nx2 arrays
		shapes       : list of image shapes
		valid_paths  : images where detection succeeded
	"""
	term = (
		cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
		30,
		1e-3
	)
	corners_all = []
	shapes = []
	valid = []

	for idx, path in enumerate(sorted(image_paths)):
		img = cv2.imread(path)
		if img is None:
			continue

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		found, corners = cv2.findChessboardCorners(
			gray,
			pattern_size,
			cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
		)
		if not found:
			print(f"[skip] image {idx+1}: corners not found -> {os.path.basename(path)}")
			continue
		corners = cv2.cornerSubPix(
			gray,
			corners,
			(11, 11),
			(-1, -1),
			term
		)
		corners_all.append(corners.reshape(-1, 2))
		shapes.append(gray.shape[:2])
		valid.append(path)

		print(f"[ok] image {idx+1} -> found {len(corners)} corners")

	return corners_all, shapes, valid


######################################
# Checkerboard World Coordinates     #
######################################
def create_world_points(pattern_size, square):
	"""Generate planar checkerboard coordinates."""
	cols, rows = pattern_size
	grid = np.zeros((cols * rows, 2))
	k = 0
	for y in range(rows):
		for x in range(cols):
			grid[k] = [x * square, y * square]
			k += 1

	return grid


###########################
# Homography Estimation   #
###########################
def estimate_homography(world, image):
	"""Direct Linear Transform (DLT)."""
	n = world.shape[0]
	A = np.zeros((2*n, 9))

	for i in range(n):
		X, Y = world[i]
		u, v = image[i]
		A[2*i] = [-X, -Y, -1, 0, 0, 0, u*X, u*Y, u]
		A[2*i+1] = [0, 0, 0, -X, -Y, -1, v*X, v*Y, v]
	_, _, vt = np.linalg.svd(A)
	H = vt[-1].reshape(3, 3)
	H /= H[2, 2]

	return H


def compute_homographies(world_pts, image_corners):
	Hs = []
	for pts in image_corners:
		Hs.append(estimate_homography(world_pts, pts))
	return Hs


######################
# Zhang Intrinsics   #
######################
def build_v(H, i, j):
	return np.array([
		H[0,i]*H[0,j],
		H[0,i]*H[1,j] + H[1,i]*H[0,j],
		H[1,i]*H[1,j],
		H[2,i]*H[0,j] + H[0,i]*H[2,j],
		H[2,i]*H[1,j] + H[1,i]*H[2,j],
		H[2,i]*H[2,j]
	])


def estimate_intrinsics(homographies):
	rows = []
	for H in homographies:
		rows.append(build_v(H,0,1))
		rows.append(build_v(H,0,0) - build_v(H,1,1))
	V = np.array(rows)
	_, _, vt = np.linalg.svd(V)
	b = vt[-1]

	B11,B12,B22,B13,B23,B33 = b
	v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
	lam = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11

	alpha = np.sqrt(abs(lam/B11))
	beta  = np.sqrt(abs(lam*B11/(B11*B22 - B12**2)))
	gamma = -B12 * alpha**2 * beta / lam
	u0    = gamma*v0/beta - B13*alpha**2/lam

	K = np.array([
		[alpha, gamma, u0],
		[0, beta, v0],
		[0, 0, 1]
	])

	with np.printoptions(precision=2, suppress=True):
		print("\nInitial intrinsic matrix:")
		print(K)

	return K


################
# Extrinsics   #
################
def estimate_extrinsics(K, Hs):
	Kinv = np.linalg.inv(K)
	poses = []

	for H in Hs:
		h1,h2,h3 = H[:,0], H[:,1], H[:,2]
		lam = 1/np.linalg.norm(Kinv @ h1)

		r1 = lam * (Kinv @ h1)
		r2 = lam * (Kinv @ h2)
		r3 = np.cross(r1,r2)
		t  = lam * (Kinv @ h3)

		Rapprox = np.column_stack((r1,r2,r3))

		U,_,Vt = np.linalg.svd(Rapprox)
		R = U@Vt
		poses.append((R,t))

	return poses


#######################
# Projection Model    #
#######################

def project_points(world, K, R, t, k):
	n = world.shape[0]
	world3 = np.hstack((world, np.zeros((n,1))))
	cam = (R @ world3.T + t.reshape(3,1)).T

	xn = cam[:,0]/cam[:,2]
	yn = cam[:,1]/cam[:,2]
	r2 = xn**2 + yn**2
	radial = 1 + k[0]*r2 + k[1]*r2**2
	xd = xn*radial
	yd = yn*radial

	fx, fy = K[0,0], K[1,1]
	cx, cy = K[0,2], K[1,2]
	s      = K[0,1]
	u = fx*xd + s*yd + cx
	v = fy*yd + cy

	return np.column_stack((u,v))


#############################
# Optimization Utilities    #
#############################
def pack_params(K, extr, k):
	p = []
	p += [K[0,0], K[1,1], K[0,2], K[1,2], K[0,1]]
	p += [k[0], k[1]]

	for R,t in extr:
		rvec,_ = cv2.Rodrigues(R)
		p += rvec.flatten().tolist()
		p += t.flatten().tolist()

	return np.array(p)


def unpack_params(p, n):
	fx,fy,cx,cy,s = p[:5]
	K = np.array([
		[fx,s,cx],
		[0,fy,cy],
		[0,0,1]
	])
	k = p[5:7]

	idx = 7
	poses = []
	for _ in range(n):
		rvec = p[idx:idx+3]
		t = p[idx+3:idx+6]
		R,_ = cv2.Rodrigues(rvec)
		poses.append((R,t))
		idx += 6

	return K,poses,k


def residuals(p, n, world, corners):
	K,extr,k = unpack_params(p,n)
	res = []
	for i in range(n):
		proj = project_points(world,K,*extr[i],k)
		diff = proj - corners[i]
		res.append(diff.flatten())

	return np.concatenate(res)


def run_optimization(K, extr, k, world, corners):
	n = len(corners)
	p0 = pack_params(K,extr,k)
	print("\nStarting nonlinear optimization...")

	result = least_squares(
		residuals,
		p0,
		args=(n,world,corners),
		method="lm",
		verbose=1
	)

	return unpack_params(result.x,n)


###########################
# Reprojection Error      #
###########################
def reprojection_error(world, corners, K, extr, k):
	errors = []

	for i in range(len(corners)):
		proj = project_points(world,K,*extr[i],k)
		d = np.linalg.norm(proj - corners[i], axis=1)
		errors.append(np.mean(d))

	return errors



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



	print('Step 4+5: Image Undistortion + Final evals')
	err = reprojection_error(world,corners,K_opt,extr_opt,k_opt)
	print("Final reprojection error:")
	print(np.mean(err))