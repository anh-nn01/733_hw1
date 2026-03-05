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



## IV. Outputs
Final intrinsics:
```
    [[2048.53   -1.83  758.73]
K = [   0.   2040.75 1345.14]
    [   0.      0.      1.  ]]
```

Final reprojection errors:
```
e = 0.58346
```