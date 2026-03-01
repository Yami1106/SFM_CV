# Buildings built in minutes - SfM 

This project implements a complete incremental Structure from Motion pipeline from scratch using Python and NumPy, reconstructing a sparse 3D point cloud of Unity Hall at WPI from a set of monocular images.

---


##  System Configuration

- **OS**: Ubuntu 24.04 LTS
- **GPU**: NVIDIA GeForce RTX 5060 Laptop
- **Python**: 3.10+

### Core Dependencies

| Library | Purpose |
|---------|---------|
| numpy | Linear algebra, matrix operations |
| scipy | Nonlinear optimization (least_squares) |
| opencv-python | Image loading and visualization |
| matplotlib | Point cloud and camera pose plotting |

---

## 📁 Folder Structure

```text
Group9_p2.zip
|   └── Phase1/
|       ├── GetInliersRANSAC.py
|       ├── EstimateFundamentalMatrix.py
|       ├── EssentialMatrixFromFundamentalMatrix.py
|       ├── ExtractCameraPose.py
|       ├── LinearTriangulation.py
|       ├── DisambiguateCameraPose.py
|       ├── NonlinearTriangulation.py
|       ├── PnPRANSAC.py
|       ├── NonlinearPnP.py
|       ├── BuildVisibilityMatrix.py
|       ├── BundleAdjustment.py
|       ├── Visualizations.py
|       ├── Wrapper.py
├──  Report.pdf
└──  README.md
```

---

##  Getting Started

### Step 1: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install numpy scipy opencv-python matplotlib
```

---

##  Running the Pipeline

Run all commands from the `Phase1/` directory.

```bash
cd Phase1/
python Wrapper.py --data_dir ../P2Data/ --calib ../P2Data/calibration.txt --num_images 5
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `../P2Data/` | Path to folder containing images and matching files |
| `--calib` | `../P2Data/calibration.txt` | Path to calibration file with intrinsic matrix K |
| `--num_images` | `5` | Number of images to process |

---