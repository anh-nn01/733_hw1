# CMSC 733 HW #1: AutoCalib
```
Name: Anh N. Nhu
Directory ID: anhu
UID: 119 385 173
```

## Overview: Zhang's Camera Calibration

This project implements Zhang's camera calibration method as described in:

> Zhengyou Zhang, ["A Flexible New Technique for Camera Calibration,"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) IEEE Transactions on Pattern Analysis and Machine Intelligence, 2000.

Specifically, provided a set of checkerboard calibration images, we:
1. Estimate the **Camera intrinsic matrix K** ($f_x, f_y, c_x, c_y$)
2. Estimate the **Radial distortion coefficients** ($k_x, k_y$)
3. Estimate the **Camera poses** (extrinsics rotation `R` and translation `t`) for each image

## Environment Setup
Use `Python 3.12` (recommended)
```sh
# install dependencies
pip install -r requirements.txt
```

## Usage guide

### Basic (default paths)
Assuming the calibration images are in `./Calibration_Imgs/`, run the following script to save the results to `./results/`
```bash
python3 Wrapper.py
```

### Custom paths
Otherwise, specify path in the following script:
```bash
python3 Wrapper.py --img_dir=<path_to_images> --out_dir=<output_directory>
```

<!-- ### Generate the PDF report

```bash
python3 generate_report.py
```

This reads results from `./results/` and generates `Report.pdf` in the project root.

## Pipeline Steps

1. **Corner Detection** — Detects 9×6 inner checkerboard corners using `cv2.findChessboardCorners` with sub-pixel refinement (`cv2.cornerSubPix`).

2. **Homography Estimation** — Computes a homography H for each image mapping world points (on the Z=0 plane, spaced 21.5mm apart) to detected image points using the Direct Linear Transform (DLT).

3. **Intrinsic Matrix Estimation** — Builds the Vb=0 linear system from all homographies (Zhang's paper Section 3.1) and solves via SVD. Extracts K from the symmetric matrix B = K^{-T} K^{-1} using the closed-form equations in Appendix B.

4. **Extrinsics Estimation** — For each image, computes R and t from K^{-1} H. The rotation matrix is corrected for orthogonality via SVD.

5. **Distortion Initialization** — Sets k1 = k2 = 0 as the initial estimate.

6. **Non-linear Optimization** — Minimizes the total geometric reprojection error over all parameters (K, all R_i, t_i, k1, k2) using Levenberg-Marquardt (`scipy.optimize.least_squares`). The distortion model follows Section 3.3 of Zhang's paper.

7. **Rectification** — Undistorts all images using the optimized K and distortion coefficients via `cv2.undistort`.

8. **Visualization** — Saves corner detection images, reprojection overlays (detected vs. reprojected corners), rectified images, and reprojection on rectified images.

## Output Structure

```
results/
├── calibration_results.txt          # Summary of all results
├── corners_detected/                # Images with detected corners drawn
│   └── corners_IMG_*.jpg
├── reprojection/                    # Original images with detected (green) and reprojected (red) corners
│   └── reproj_IMG_*.jpg
├── rectified/                       # Undistorted (rectified) images
│   └── rect_IMG_*.jpg
└── reproj_on_rectified/             # Rectified images with reprojected corners
    └── reproj_rect_IMG_*.jpg
```

## Checkerboard Specifications

- Pattern: 10×7 squares (9×6 inner corners)
- Square size: 21.5 mm
- Printed on A4 paper
- X-axis: even number of squares, Y-axis: odd number of squares

## Results Summary

| Parameter | Value |
|-----------|-------|
| fx | 2048.53 |
| fy | 2040.75 |
| cx | 758.73 |
| cy | 1345.14 |
| k1 | 0.1731 |
| k2 | -0.7534 |
| Mean Reprojection Error | 0.5835 px | -->