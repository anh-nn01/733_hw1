"""
Author: Anh N. Nhu
UID: 119 385 173

Auto Calibration:

Steps:
# 1. Checkerboard corner detection
# 2. Per-image homography estimation (DLT)
# 3. Intrinsic matrix (K) using Zhang's method
# 4. Extrinsic parameter estimation (R, t) per image
# 5. Radial distortion initialization (k1 = k2 = 0)
# 6. Joint non-linear optimization (scipy.least_squares)
# 7. Image undistortion
# 8. Reprojection error computation and visualization
"""

import os
import sys
import argparse
import glob
import numpy as np
from scipy.optimize import least_squares
import cv2
import matplotlib
import matplotlib.pyplot as plt




#############################
# Checkerboard parameters   #
#############################
CHECKER_ROWS = 6        
CHECKER_COLS = 9     
SQUARE_SIZE  = 21.5  # given


# CMD arguments
parser = argparse.ArgumentParser(description="Zhang's Camera Calibration")
parser.add_argument('--img_dir', type=str, default='Calibration_Imgs',
					help='Directory containing calibration images')
parser.add_argument('--out_dir', type=str, default='results',
					help='Output directory for results')
args = parser.parse_args()
matplotlib.use('Agg')



#############################
# 1. Corner Detection
#############################
def detect_corners(img_paths, pattern_size=(CHECKER_COLS, CHECKER_ROWS)):
	"""
	Detect checkerboard corners (cv2.findChessboardCorners).

	Outputs:
		all_img_corners: list[(N, 2)]: list of detected corners in all images
		img_shapes:  list[(h, w)]: list of all image shapes
		accepted_paths: acceptable image paths with successful detections
	"""
	all_img_corners = []
	all_img_shapes = []
	accepted_paths = []
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	for idx, img_path in enumerate(sorted(img_paths)):
		if img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('jpeg'):
			img = cv2.imread(img_path)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# corner detection
			ret, corners = cv2.findChessboardCorners(
				gray, pattern_size,
				cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
			)
			# make sure there is at least a corner
			if not ret:
				print(f"Image {idx+1}:\t detect 0 corners!")
				continue
			
			# improve corners
			corners_refined = cv2.cornerSubPix(
				gray, corners, (11, 11), (-1, -1), criteria
			)
			all_img_corners.append(corners_refined.reshape(-1, 2))
			all_img_shapes.append(gray.shape[:2])
			accepted_paths.append(img_path)
			print(f"Image {idx+1}:\t detect {len(corners_refined)} corners!")

	return all_img_corners, all_img_shapes, accepted_paths


# ============================================================
# 2. World Points
# ============================================================
def get_world_points(pattern_size=(CHECKER_COLS, CHECKER_ROWS), square_size=SQUARE_SIZE):
	"""Generate world coordinates for checkerboard corners on Z=0 plane.
	Points are spaced by square_size (21.5 mm)."""
	pts = np.zeros((pattern_size[0] * pattern_size[1], 2), dtype=np.float64)
	for i in range(pattern_size[1]):
		for j in range(pattern_size[0]):
			pts[i * pattern_size[0] + j] = [j * square_size, i * square_size]
	return pts


# ============================================================
# 3. Homography Estimation (DLT)
# ============================================================
def compute_homography(world_pts, img_pts):
	"""Compute homography from world (X,Y) to image (u,v) using Direct Linear
	Transform (DLT). Solves Ah=0 via SVD.

	world_pts: (N, 2), img_pts: (N, 2)
	Returns H: (3, 3)
	"""
	N = world_pts.shape[0]
	A = np.zeros((2 * N, 9), dtype=np.float64)
	for i in range(N):
		X, Y = world_pts[i]
		u, v = img_pts[i]
		A[2 * i]     = [-X, -Y, -1, 0, 0, 0, u*X, u*Y, u]
		A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, v*X, v*Y, v]

	_, _, Vt = np.linalg.svd(A)
	H = Vt[-1].reshape(3, 3)
	H /= H[2, 2]
	return H


def compute_all_homographies(world_pts, all_corners):
	"""Compute homography for each image."""
	homographies = []
	for corners in all_corners:
		H = compute_homography(world_pts, corners)
		homographies.append(H)
	return homographies


# ============================================================
# 4. Zhang's Closed-Form K Estimation
# ============================================================
def v_ij(H, i, j):
	"""Construct the v_ij vector from homography H.
	Reference: Zhang's paper, Section 3.1, Equation (4)."""
	return np.array([
		H[0, i] * H[0, j],
		H[0, i] * H[1, j] + H[1, i] * H[0, j],
		H[1, i] * H[1, j],
		H[2, i] * H[0, j] + H[0, i] * H[2, j],
		H[2, i] * H[1, j] + H[1, i] * H[2, j],
		H[2, i] * H[2, j]
	])


def estimate_intrinsics(homographies):
	"""Estimate K from homographies using Zhang's closed-form solution.
	Builds the Vb=0 system from all homographies and solves via SVD.
	Then extracts K from the symmetric matrix B = K^{-T} K^{-1}.
	Reference: Zhang's paper, Section 3.1 and Appendix B.
	"""
	V = []
	for H in homographies:
		V.append(v_ij(H, 0, 1))
		V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
	V = np.array(V)

	_, _, Vt = np.linalg.svd(V)
	b = Vt[-1]

	# b = [B11, B12, B22, B13, B23, B33]
	B11, B12, B22, B13, B23, B33 = b

	# Extract intrinsic parameters (Zhang's paper Appendix B)
	v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
	lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
	alpha = np.sqrt(abs(lam / B11))
	beta = np.sqrt(abs(lam * B11 / (B11 * B22 - B12**2)))
	gamma = -B12 * alpha**2 * beta / lam
	u0 = gamma * v0 / beta - B13 * alpha**2 / lam

	K = np.array([
		[alpha, gamma, u0],
		[0,     beta,  v0],
		[0,     0,     1]
	], dtype=np.float64)

	print(f"\n[INFO] Initial K (closed-form):")
	print(K)
	return K


# ============================================================
# 5. Extrinsics Estimation
# ============================================================
def estimate_extrinsics(K, homographies):
	"""Estimate R, t for each image from K and H.
	Reference: Zhang's paper, Section 3.1.
	r1 = lambda * K^{-1} h1, r2 = lambda * K^{-1} h2, r3 = r1 x r2, t = lambda * K^{-1} h3
	Rotation matrix is approximated via SVD to enforce orthogonality.
	"""
	K_inv = np.linalg.inv(K)
	extrinsics = []

	for H in homographies:
		h1 = H[:, 0]
		h2 = H[:, 1]
		h3 = H[:, 2]

		lam = 1.0 / np.linalg.norm(K_inv @ h1)

		r1 = lam * (K_inv @ h1)
		r2 = lam * (K_inv @ h2)
		r3 = np.cross(r1, r2)
		t  = lam * (K_inv @ h3)

		# Approximate rotation matrix via SVD to enforce orthogonality
		Q = np.column_stack([r1, r2, r3])
		U, _, Vt = np.linalg.svd(Q)
		R = U @ Vt

		extrinsics.append((R, t))

	return extrinsics


# ============================================================
# 6. Projection and Distortion Model
# ============================================================
def project_points(world_pts, K, R, t, k):
	"""Project world points to image using K, R, t and radial distortion k=[k1,k2].
	Distortion model (Zhang's paper Section 3.3):
		x_d = x_n * (1 + k1*r^2 + k2*r^4)
	where x_n are normalized (ideal) image coordinates and r^2 = x_n^2 + y_n^2.

	world_pts: (N, 2) — X, Y on Z=0 plane
	Returns: (N, 2) projected image points
	"""
	N = world_pts.shape[0]
	X_w = np.hstack([world_pts, np.zeros((N, 1))])  # (N, 3)

	# Transform to camera frame
	X_c = (R @ X_w.T + t.reshape(3, 1)).T  # (N, 3)

	# Normalized image coordinates
	x_n = X_c[:, 0] / X_c[:, 2]
	y_n = X_c[:, 1] / X_c[:, 2]

	# Radial distortion
	r2 = x_n**2 + y_n**2
	radial = 1 + k[0] * r2 + k[1] * r2**2

	x_d = x_n * radial
	y_d = y_n * radial

	# Apply K
	fx, fy = K[0, 0], K[1, 1]
	cx, cy = K[0, 2], K[1, 2]
	gamma  = K[0, 1]

	u = fx * x_d + gamma * y_d + cx
	v = fy * y_d + cy

	return np.column_stack([u, v])


# ============================================================
# 7. Non-linear Optimization
# ============================================================
def pack_params(K, extrinsics, k):
	"""Pack all parameters into a single vector for optimization.
	Layout: [fx, fy, cx, cy, gamma, k1, k2, rvec1(3), t1(3), rvec2(3), t2(3), ...]
	"""
	params = []
	params.extend([K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]])
	params.extend([k[0], k[1]])
	for R, t in extrinsics:
		rvec, _ = cv2.Rodrigues(R)
		params.extend(rvec.flatten())
		params.extend(t.flatten())
	return np.array(params, dtype=np.float64)


def unpack_params(params, n_images):
	"""Unpack parameter vector back into K, extrinsics, k."""
	fx, fy, cx, cy, gamma = params[0:5]
	K = np.array([
		[fx, gamma, cx],
		[0,  fy,    cy],
		[0,  0,     1]
	], dtype=np.float64)

	k = params[5:7]

	extrinsics = []
	idx = 7
	for _ in range(n_images):
		rvec = params[idx:idx+3]
		t = params[idx+3:idx+6]
		R, _ = cv2.Rodrigues(rvec)
		extrinsics.append((R, t))
		idx += 6

	return K, extrinsics, k


def residuals(params, n_images, world_pts, all_corners):
	"""Compute residual vector (predicted - observed) for all images and all points.
	This is the geometric error that we minimize:
	sum_i sum_j || x_{i,j} - x_hat_{i,j}(K, R_i, t_i, X_j, k) ||
	"""
	K, extrinsics, k = unpack_params(params, n_images)
	res = []
	for i in range(n_images):
		R, t = extrinsics[i]
		projected = project_points(world_pts, K, R, t, k)
		diff = projected - all_corners[i]
		res.append(diff.flatten())
	return np.concatenate(res)


def optimize(K, extrinsics, k, world_pts, all_corners):
	"""Run Levenberg-Marquardt non-linear least squares optimization
	using scipy.optimize.least_squares."""
	n_images = len(all_corners)
	p0 = pack_params(K, extrinsics, k)

	print(f"\n[INFO] Starting non-linear optimization with {len(p0)} parameters...")
	result = least_squares(
		residuals, p0,
		args=(n_images, world_pts, all_corners),
		method='lm',
		verbose=1
	)
	print(f"[INFO] Optimization finished. Cost: {result.cost:.4f}")

	K_opt, ext_opt, k_opt = unpack_params(result.x, n_images)
	return K_opt, ext_opt, k_opt


# ============================================================
# 8. Reprojection Error
# ============================================================
def compute_reprojection_error(world_pts, all_corners, K, extrinsics, k):
	"""Compute per-image mean reprojection error (Euclidean distance in pixels)."""
	errors = []
	for i in range(len(all_corners)):
		R, t = extrinsics[i]
		projected = project_points(world_pts, K, R, t, k)
		err = np.sqrt(np.sum((projected - all_corners[i])**2, axis=1))
		mean_err = np.mean(err)
		errors.append(mean_err)
	return errors


# ============================================================
# 9. Visualization
# ============================================================
def save_corner_detection(img_paths, all_corners, out_dir, pattern_size=(CHECKER_COLS, CHECKER_ROWS)):
	"""Save images with detected corners drawn on them."""
	corner_dir = os.path.join(out_dir, 'corners_detected')
	os.makedirs(corner_dir, exist_ok=True)

	for i, path in enumerate(img_paths):
		img = cv2.imread(path)
		corners_cv = all_corners[i].reshape(-1, 1, 2).astype(np.float32)
		cv2.drawChessboardCorners(img, pattern_size, corners_cv, True)
		fname = os.path.join(corner_dir, f"corners_{os.path.basename(path)}")
		cv2.imwrite(fname, img)
	print(f"[INFO] Saved corner detection images to {corner_dir}")


def save_rectified_images(img_paths, K, k, out_dir):
	"""Undistort (rectify) images using estimated K and distortion coefficients.
	Uses cv2.undistort with dist_coeffs = [k1, k2, 0, 0, 0]."""
	rect_dir = os.path.join(out_dir, 'rectified')
	os.makedirs(rect_dir, exist_ok=True)

	dist_coeffs = np.array([k[0], k[1], 0.0, 0.0, 0.0])

	for path in img_paths:
		img = cv2.imread(path)
		undistorted = cv2.undistort(img, K, dist_coeffs)
		fname = os.path.join(rect_dir, f"rect_{os.path.basename(path)}")
		cv2.imwrite(fname, undistorted)
	print(f"[INFO] Saved rectified (undistorted) images to {rect_dir}")


def save_reprojection_on_rectified(img_paths, all_corners, world_pts, K, extrinsics, k, out_dir):
	"""Draw reprojected corners on rectified (undistorted) images."""
	reproj_rect_dir = os.path.join(out_dir, 'reproj_on_rectified')
	os.makedirs(reproj_rect_dir, exist_ok=True)

	dist_coeffs = np.array([k[0], k[1], 0.0, 0.0, 0.0])
	k_zero = np.array([0.0, 0.0])  # no distortion for reprojection on undistorted image

	for i, path in enumerate(img_paths):
		img = cv2.imread(path)
		undistorted = cv2.undistort(img, K, dist_coeffs)

		R, t = extrinsics[i]
		# Project without distortion (since image is already undistorted)
		projected = project_points(world_pts, K, R, t, k_zero)

		# Undistort the detected corners too
		corners_undist = cv2.undistortPoints(
			all_corners[i].reshape(-1, 1, 2).astype(np.float32),
			K, dist_coeffs, P=K
		).reshape(-1, 2)

		# Draw undistorted detected corners (green circles)
		for pt in corners_undist:
			cv2.circle(undistorted, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)

		# Draw reprojected corners (red filled circles)
		for pt in projected:
			cv2.circle(undistorted, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

		fname = os.path.join(reproj_rect_dir, f"reproj_rect_{os.path.basename(path)}")
		cv2.imwrite(fname, undistorted)
	print(f"[INFO] Saved reprojection on rectified images to {reproj_rect_dir}")


def visualize_reprojection(img_paths, all_corners, world_pts, K, extrinsics, k, out_dir):
	"""Draw detected (green) and reprojected (red) corners on original images."""
	reproj_dir = os.path.join(out_dir, 'reprojection')
	os.makedirs(reproj_dir, exist_ok=True)

	for i, path in enumerate(img_paths):
		img = cv2.imread(path)
		R, t = extrinsics[i]
		projected = project_points(world_pts, K, R, t, k)
		detected = all_corners[i]

		# Draw detected corners (green)
		for pt in detected:
			cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)

		# Draw reprojected corners (red)
		for pt in projected:
			cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

		fname = os.path.join(reproj_dir, f"reproj_{os.path.basename(path)}")
		cv2.imwrite(fname, img)
	print(f"[INFO] Saved reprojection images to {reproj_dir}")


if __name__ == '__main__':
	# Gather image paths
	img_paths = sorted(
		glob.glob(os.path.join(args.img_dir, '*.jpg')) +
		glob.glob(os.path.join(args.img_dir, '*.png'))
	)
	if not img_paths:
		print("[ERROR] No images found in", args.img_dir)
		sys.exit(1)
	print(f"[INFO] Found {len(img_paths)} images in {args.img_dir}")

	os.makedirs(args.out_dir, exist_ok=True)

	# ---- Step 1: Detect corners ----
	print("\n" + "="*60)
	print("Step 1: Corner Detection")
	print("="*60)
	all_corners, img_shapes, valid_paths = detect_corners(img_paths)
	if len(all_corners) < 3:
		print("[ERROR] Need at least 3 images with detected corners.")
		sys.exit(1)
	print(f"[INFO] Successfully detected corners in {len(all_corners)} images")

	# Save corner detection visualization
	save_corner_detection(valid_paths, all_corners, args.out_dir)

	# ---- Step 2: World points ----
	world_pts = get_world_points()

	# ---- Step 3: Compute homographies ----
	print("\n" + "="*60)
	print("Step 2: Homography Estimation")
	print("="*60)
	homographies = compute_all_homographies(world_pts, all_corners)
	print(f"[INFO] Computed {len(homographies)} homographies")

	# ---- Step 4: Estimate K (closed-form) ----
	print("\n" + "="*60)
	print("Step 3: Intrinsic Matrix Estimation (Zhang's Closed-Form)")
	print("="*60)
	K = estimate_intrinsics(homographies)

	# ---- Step 5: Estimate extrinsics ----
	print("\n" + "="*60)
	print("Step 4: Extrinsics Estimation")
	print("="*60)
	extrinsics = estimate_extrinsics(K, homographies)
	print(f"[INFO] Estimated extrinsics for {len(extrinsics)} images")

	# ---- Step 6: Initial distortion ----
	k_init = np.array([0.0, 0.0])

	# Reprojection error before optimization
	print("\n" + "="*60)
	print("Initial Reprojection Error (before optimization)")
	print("="*60)
	errors_init = compute_reprojection_error(world_pts, all_corners, K, extrinsics, k_init)
	for i, err in enumerate(errors_init):
		print(f"  Image {i+1:2d} ({os.path.basename(valid_paths[i])}): {err:.4f} px")
	print(f"  Mean: {np.mean(errors_init):.4f} px")

	# ---- Step 7: Non-linear optimization ----
	print("\n" + "="*60)
	print("Step 5: Non-linear Geometric Error Minimization")
	print("="*60)
	K_opt, ext_opt, k_opt = optimize(K, extrinsics, k_init, world_pts, all_corners)

	# ---- Final results ----
	print("\n" + "="*60)
	print("FINAL RESULTS")
	print("="*60)
	print(f"\nOptimized Camera Intrinsic Matrix K:")
	print(K_opt)
	print(f"\nDistortion Coefficients:")
	print(f"  k1 = {k_opt[0]:.6f}")
	print(f"  k2 = {k_opt[1]:.6f}")

	# Reprojection error after optimization
	print("\n" + "-"*60)
	print("Final Reprojection Error (after optimization)")
	print("-"*60)
	errors_opt = compute_reprojection_error(world_pts, all_corners, K_opt, ext_opt, k_opt)
	for i, err in enumerate(errors_opt):
		print(f"  Image {i+1:2d} ({os.path.basename(valid_paths[i])}): {err:.4f} px")
	mean_err = np.mean(errors_opt)
	print(f"  Mean: {mean_err:.4f} px")

	# ---- Step 8: Visualization ----
	print("\n" + "="*60)
	print("Step 6: Saving Visualizations")
	print("="*60)

	# Reprojection on original images
	visualize_reprojection(valid_paths, all_corners, world_pts, K_opt, ext_opt, k_opt, args.out_dir)

	# Rectified (undistorted) images
	save_rectified_images(valid_paths, K_opt, k_opt, args.out_dir)

	# Reprojection on rectified images
	save_reprojection_on_rectified(valid_paths, all_corners, world_pts, K_opt, ext_opt, k_opt, args.out_dir)

	# ---- Save summary text file ----
	summary_path = os.path.join(args.out_dir, 'calibration_results.txt')
	with open(summary_path, 'w') as f:
		f.write("Zhang's Camera Calibration Results\n")
		f.write("=" * 60 + "\n\n")

		f.write("Initial Camera Intrinsic Matrix K (closed-form):\n")
		f.write(np.array2string(K, precision=4, suppress_small=True) + "\n\n")

		f.write("Optimized Camera Intrinsic Matrix K:\n")
		f.write(np.array2string(K_opt, precision=4, suppress_small=True) + "\n\n")

		f.write(f"Distortion Coefficients:\n")
		f.write(f"  k1 = {k_opt[0]:.6f}\n")
		f.write(f"  k2 = {k_opt[1]:.6f}\n\n")

		f.write("Per-Image Reprojection Error:\n")
		f.write(f"{'Image':<8} {'Filename':<35} {'Before (px)':<15} {'After (px)':<15}\n")
		f.write("-" * 73 + "\n")
		for i in range(len(valid_paths)):
			f.write(f"{i+1:<8} {os.path.basename(valid_paths[i]):<35} "
					f"{errors_init[i]:<15.4f} {errors_opt[i]:<15.4f}\n")
		f.write("-" * 73 + "\n")
		f.write(f"{'Mean':<8} {'':<35} {np.mean(errors_init):<15.4f} {mean_err:<15.4f}\n\n")

		f.write("Extrinsics per image:\n")
		for i, (R, t) in enumerate(ext_opt):
			f.write(f"\nImage {i+1} ({os.path.basename(valid_paths[i])}):\n")
			f.write(f"  R:\n{np.array2string(R, precision=6)}\n")
			f.write(f"  t: {np.array2string(t, precision=6)}\n")

	print(f"[INFO] Results saved to {summary_path}")
	print("\n[DONE] Camera calibration complete.")
