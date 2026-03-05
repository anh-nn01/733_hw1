# CMSC 733 HW #1: AutoCalib
```
Name: Anh N. Nhu
Directory ID: anhu
UID: 119 385 173
```

## I. Overview: Zhang's Camera Calibration

This project implements Zhang's camera calibration method:

Zhengyou Zhang: ["A Flexible New Technique for Camera Calibration,"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf).

Specifically, provided a set of checkerboard calibration images, we:
1. Estimate the **Camera intrinsic matrix K** ($f_x, f_y, c_x, c_y$)
2. Estimate the **Radial distortion coefficients** ($k_x, k_y$)
3. Estimate the **Camera poses** (extrinsics rotation `R` and translation `t`) for each image

## II. Environment Setup
Use `Python 3.12` (recommended)
```sh
# install dependencies
pip install -r requirements.txt
```

## III. Usage guide

### a. Basic (default paths)
Assuming the calibration images are in `./Calibration_Imgs/`, run the following script to save the results to `./results/`
```bash
python3 Wrapper.py
```

### b. Custom image paths
Otherwise, specify path in the following script:
```bash
python3 Wrapper.py --img_dir=<path_to_images>
```

<!-- ## Output Structure

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

## Results Summary

| Parameter | Value |
|-----------|-------|
| fx | 2048.53 |
| fy | 2040.75 |
| cx | 758.73 |
| cy | 1345.14 |
| k1 | 0.1731 |
| k2 | -0.7534 |
| Mean Reprojection Error | 0.5835 px |  -->