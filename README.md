<div align="center">

# Structure from Motion — 3D Reconstruction from Photos

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)

*Reconstructing the 3D shape of a real building from 5 ordinary photographs — built entirely from scratch.*

</div>

---

## What it does

Takes a set of overlapping images of a scene and recovers the full 3D point cloud along with the camera positions that took them. Every stage — feature matching, geometry estimation, triangulation, and bundle adjustment — is implemented from scratch without calling high-level SfM libraries.

Applied to 5 images of **Unity Hall at WPI** to produce a dense 3D reconstruction.

---

## Pipeline

```
Images → SIFT Features → Feature Matching
       → Fundamental Matrix (Hartley + SVD rank-2)
       → RANSAC outlier rejection
       → Essential Matrix → Camera Pose (cheirality check)
       → Linear Triangulation → Nonlinear Refinement
       → PnP + RANSAC (new cameras)
       → Bundle Adjustment → Final 3D Point Cloud
```

---

## Results

| Stage | Detail |
|---|---|
| Feature matching | SIFT across all image pairs |
| Best inlier ratio | 96% (1137 / 1183 matches, pair 3–4) |
| Points triangulated | 592 (linear) → 562 (nonlinear refined) |
| Bundle adjustment | 1318 points across 4 cameras |
| Reprojection error | Significantly reduced by nonlinear refinement |

> Camera 3 was automatically rejected due to insufficient inliers — demonstrating robust PnP behaviour.

---

## Key concepts implemented

- **Hartley normalisation** for numerical stability in DLT
- **SVD rank-2 enforcement** on the fundamental matrix
- **Cheirality check** to resolve the 4 ambiguous poses from essential matrix decomposition
- **Linear triangulation** (DLT) followed by **nonlinear minimisation** of reprojection error
- **PnP + RANSAC** to register additional cameras into the world frame
- **Bundle adjustment** (Levenberg–Marquardt) for global consistency

---

## Tech stack

`Python` · `NumPy` · `OpenCV` · `SciPy`

---

<div align="center">
Part of the WPI Computer Vision course · <a href="https://github.com/Yami1106">Ashish Sukumar</a>
</div>
