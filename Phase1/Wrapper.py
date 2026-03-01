"""
Wrapper.py — Full Structure from Motion pipeline.
All visualizations saved to ./outputs/ using Visualizations.py.

Usage:
  python Wrapper.py --data_dir Data/ --calib Data/calibration.txt --num_images 5
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from EstimateFundamentalMatrix            import EstimateFundamentalMatrix, make_homogeneous
from GetInliersRANSAC                     import GetInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import EfromF
from ExtractCameraPose                    import ExtractCameraPose
from LinearTriangulation                  import LinearTriangulation, camera_projection_matrix
from DisambiguateCameraPose               import DisambiguateCameraPose
from NonlinearTriangulation               import NonlinearTriangulation
from LinearPnP                            import LinearPnP
from PnPRANSAC                            import PnPRANSAC
from NonlinearPnP                         import NonlinearPnP
from BuildVisibilityMatrix                import BuildVisibilityMatrix
from BundleAdjustment                     import BundleAdjustment

from Visualizations import (
    save_matches_ransac,
    save_reprojection_overlay,
    save_pointcloud_xz,
    save_pointcloud_compare_xz,
    save_candidate_poses_xz,
    save_pnp_compare_xz,
    save_ba_before_after_xz,
    save_final_reconstruction,
)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# this function is used to filter outliers from triangulated points  after each new camera is added and BA is run
def filter_outliers_with_mask(X: np.ndarray, percentile: int = 95):
    """
    Inputs:
      X          : (N,3) array of 3D points
      percentile : distance threshold — points beyond this
                   distance from the median are removed
    Returns:
      X_filtered : (M,3) array of inlier 3D points where M <= N
      keep       : (N,) boolean array where True means the point was kept
    """

    if len(X) == 0:
        return X, np.ones(0, dtype=bool)
    median = np.median(X, axis=0)
    dists  = np.linalg.norm(X - median, axis=1)
    thresh = np.percentile(dists, percentile)
    keep   = dists < thresh
    return X[keep], keep


# 
def filter_outliers(X: np.ndarray, percentile: int = 95) -> np.ndarray:
    """
    Inputs:
      X          : (N,3) array of 3D points
      percentile : distance threshold — points beyond this
                   distance from the median are removed
    Returns:
      X_filtered : (M,3) array of inlier 3D points where M <= N
    """

    X_f, _ = filter_outliers_with_mask(X, percentile)
    return X_f

# this function is used to filter both 3D points and tracks after each new camera is added and BA is run
def filter_tracks_and_points(X: np.ndarray, tracks: list, percentile: int = 90):
    """
    Inputs:
      X          : (N,3) array of 3D points
      tracks     : list of N dicts where tracks[j][i] = (u,v) is the
                   observed pixel of point j in camera i
      percentile : distance threshold — points beyond this
                   distance from the median are removed
    Returns:
      X_filtered      : (M,3) array of inlier 3D points where M <= N
      tracks_filtered : list of M dicts corresponding to the kept points,
                        synchronized with X_filtered
    """

    X_filtered, keep = filter_outliers_with_mask(X, percentile)
    tracks_filtered  = [tracks[j] for j in range(len(tracks)) if keep[j]]
    return X_filtered, tracks_filtered


# get the camera intrinsics from the calibration file
def load_intrinsics(calib_path: str) -> np.ndarray:
    return np.loadtxt(calib_path)

# load the images using OpenCV
def load_image(data_dir: str, idx: int):
    for ext in (".jpg", ".png", ".JPG", ".PNG"):
        p = os.path.join(data_dir, f"{idx+1}{ext}")
        if os.path.exists(p):
            return cv2.imread(p)
    return None

# load the matches form the matching.txt file
def load_matches(data_dir: str):
    import glob, re
    matches = {}

    for path in sorted(glob.glob(os.path.join(data_dir, "matching*.txt"))):
        base = int(re.search(r'matching(\d+)', path).group(1)) - 1

        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('nFeatures'):
                continue

            vals      = line.split()
            n_matches = int(vals[0])
            u1        = float(vals[4])
            v1        = float(vals[5])

            idx = 6
            for _ in range(n_matches):
                if idx + 2 >= len(vals):
                    break
                target = int(vals[idx]) - 1
                u2     = float(vals[idx + 1])
                v2     = float(vals[idx + 2])
                idx   += 3

                if target < 0:
                    continue

                key = (base, target)
                if key not in matches:
                    matches[key] = ([], [])
                matches[key][0].append([u1, v1])
                matches[key][1].append([u2, v2])

    return {
        k: (np.array(v[0]), np.array(v[1]))
        for k, v in matches.items()
    }

# this is used to reproject 3D points into a camera for visualization
def reproject(K, C, R, X) -> np.ndarray:
    """
    Inputs:
      K : (3,3) intrinsic camera matrix
      C : (3,)  camera center in world coordinates
      R : (3,3) rotation matrix
      X : (N,3) 3D world points
    Returns:
      x : (N,2) projected 2D pixel coordinates
    """

    P   = camera_projection_matrix(K, C, R)
    X_h = make_homogeneous(X)
    x_h = (P @ X_h.T).T
    return x_h[:, :2] / x_h[:, 2:3]


def get_pnp_correspondences(cam_i, X, tracks):
    """
    Inputs:
      cam_i  : integer index of the camera being registered
      X      : (N,3) array of current 3D world points
      tracks : list of N dicts where tracks[j][i] = (u,v) is the
               observed pixel of point j in camera i
    Returns:
      X_list : (M,3) array of 3D points visible in camera cam_i
      x_list : (M,2) array of corresponding 2D pixel observations
               in camera cam_i, where M <= N
    """

    X_list, x_list = [], []
    for j, track in enumerate(tracks):
        if cam_i in track:
            if j >= len(X):
                continue
            X_list.append(X[j])
            x_list.append(track[cam_i])
    if not X_list:
        return np.zeros((0, 3)), np.zeros((0, 2))
    return np.array(X_list), np.array(x_list)


def tracks_for_new_camera(cam_i, tracks, clean):
    """
    Inputs:
      cam_i  : integer index of the new camera being registered
      tracks : list of N dicts where tracks[j][i] = (u,v) is the
               observed pixel of point j in camera i 
      clean  : dict mapping image pair (i,j) to (x_i, x_j) arrays
               of (N,2) RANSAC-filtered pixel correspondences
    """

    count = 0
    for j, track in enumerate(tracks):
        for cam_k in list(track.keys()):
            key_fwd = (cam_k, cam_i)
            key_rev = (cam_i, cam_k)

            if key_fwd in clean:
                x_k, x_i = clean[key_fwd]
                obs_k = np.array(track[cam_k])
                dists = np.linalg.norm(x_k - obs_k, axis=1)
                best  = np.argmin(dists)
                if dists[best] < 2.0:
                    track[cam_i] = tuple(x_i[best])
                    count += 1
                    break

            elif key_rev in clean:
                x_i, x_k = clean[key_rev]
                obs_k = np.array(track[cam_k])
                dists = np.linalg.norm(x_k - obs_k, axis=1)
                best  = np.argmin(dists)
                if dists[best] < 2.0:
                    track[cam_i] = tuple(x_i[best])
                    count += 1
                    break

    print(f"  Populated {count} track observations for camera {cam_i+1}")


def find_best_ref(cam_i, registered, clean):

    """
    Inputs:
      cam_i      : integer index of the new camera being registered
      registered : list of integer indices of already registered cameras
      clean      : dict mapping image pair (i,j) to (x_i, x_j) arrays
                   of (N,2) RANSAC-filtered pixel correspondences
    Returns:
      best_ref   : integer index of the already registered camera that
                   has the highest number of clean correspondences with
                   cam_i — used as the triangulation reference camera
    """

    best_ref, best_count = 0, 0
    for k in registered:
        key_fwd = (k, cam_i)
        key_rev = (cam_i, k)
        if key_fwd in clean:
            count = len(clean[key_fwd][0])
        elif key_rev in clean:
            count = len(clean[key_rev][0])
        else:
            count = 0
        if count > best_count:
            best_count = count
            best_ref   = k
    return best_ref



def run_sfm(data_dir: str, calib_file: str, num_images: int):
    """
    Inputs:
      data_dir   : path to directory containing input images and
                   matching txt files
      calib_file : path to calibration file containing the (3,3)
                   intrinsic matrix K
      num_images : integer number of images to process
    Returns:
      Cset       : list of I refined camera centers, each (3,),
                   one per successfully registered camera
      Rset       : list of I refined rotation matrices, each (3,3),
                   one per successfully registered camera
      X          : (J,3) array of final refined 3D world points
                   after bundle adjustment and outlier filtering
    """

    print("=" * 60)
    print("Structure from Motion — Full Pipeline")
    print("=" * 60)

    K       = load_intrinsics(calib_file)
    matches = load_matches(data_dir)
    images  = [load_image(data_dir, i) for i in range(num_images)]

    np.save(os.path.join(OUT, "matches_raw.npy"), matches)

    # 1): RANSAC filter all pairs
    print("\n[1] Filtering correspondences with RANSAC...")
    clean = {}
    for (i, j), (x_i, x_j) in matches.items():
        mask = GetInliersRANSAC(x_i, x_j)
        if mask.shape[0] != x_i.shape[0]:
            print(f"  WARNING: mask length mismatch for pair ({i+1},{j+1}) - skipping")
            clean[(i,j)] = (np.zeros((0,2)), np.zeros((0,2)))
            continue

        clean[(i,j)] = (x_i[mask], x_j[mask])
        print(f"  Pair ({i+1},{j+1}): {mask.sum()} / {len(mask)} inliers")

        if images[i] is not None and images[j] is not None:
            out_path = os.path.join(OUT, f"matches_{i+1}_{j+1}.png")
            save_matches_ransac(
                images[i], images[j], x_i, x_j, mask,
                out_path=out_path
            )
            print(f"  Saved {out_path}")

    np.save(os.path.join(OUT, "matches_clean.npy"), clean)

    # 2): Bootstrap — images 1 & 2 
    print("\n[2] Bootstrapping from images 1 & 2...")
    if (0, 1) not in clean or len(clean[(0,1)][0]) < 8:
        raise RuntimeError("Not enough inliers between images 1 and 2 to bootstrap (need >=8)")

    x1, x2 = clean[(0, 1)]

    F  = EstimateFundamentalMatrix(x1, x2)
    np.save(os.path.join(OUT, "F_1_2.npy"), F)
    E  = EfromF(F, K)
    np.save(os.path.join(OUT, "E_1_2.npy"), E)

    Cset_cand, Rset_cand = ExtractCameraPose(E)

    C0, R0    = np.zeros(3), np.eye(3)
    Xset_cand = [
        LinearTriangulation(K, C0, R0, Cset_cand[k], Rset_cand[k], x1, x2)
        for k in range(4)
    ]

    save_candidate_poses_xz(
        Xset_cand,
        out_path=os.path.join(OUT, "candidate_poses_xz.png"),
        title="initial triangulation"
    )
    for k in range(4):
        np.save(os.path.join(OUT, f"X_candidate_k{k+1}.npy"), Xset_cand[k])

    C2, R2, X_lin = DisambiguateCameraPose(K, Cset_cand, Rset_cand, x1, x2)
    print(f"  Pose disambiguated — {len(X_lin)} points triangulated")

    np.save(os.path.join(OUT, "C_bootstrap.npy"), np.vstack([C0, C2]))
    np.save(os.path.join(OUT, "R_bootstrap.npy"), np.stack([R0, R2]))

    X_nonlin = NonlinearTriangulation(K, C0, R0, C2, R2, x1, x2, X_lin)
    np.save(os.path.join(OUT, "X_nonlinear_init.npy"), X_nonlin)

    X_nonlin = filter_outliers(X_nonlin)
    print(f"  After outlier filter — {len(X_nonlin)} points remaining")
    np.save(os.path.join(OUT, "X_nonlinear_init_filtered.npy"), X_nonlin)

    save_pointcloud_compare_xz(
        X_lin, X_nonlin,
        out_path=os.path.join(OUT, "linear_vs_nonlinear_1_2.png"),
        title="Linear vs Nonlinear — images 1 & 2",
        Cset=[C0, C2],
        Rset=[R0, R2]
    )
    print(f"  Saved {os.path.join(OUT, 'linear_vs_nonlinear_1_2.png')}")

    for cam_idx, (C, R, x_obs) in enumerate([(C0, R0, x1), (C2, R2, x2)]):
        real_idx = [0, 1][cam_idx]
        if images[real_idx] is not None:
            out_lin = os.path.join(OUT, f"reproj_linear_cam{real_idx+1}.png")
            save_reprojection_overlay(
                images[real_idx], x_obs,
                reproject(K, C, R, X_lin),
                out_path=out_lin,
                title=f"lin reproj vis, frame {real_idx}"
            )
            print(f"  Saved {out_lin}")

            out_nonlin = os.path.join(OUT, f"reproj_nonlinear_cam{real_idx+1}.png")
            save_reprojection_overlay(
                images[real_idx], x_obs,
                reproject(K, C, R, X_nonlin),
                out_path=out_nonlin,
                title=f"nonlin reproj vis, frame {real_idx}"
            )
            print(f"  Saved {out_nonlin}")

    # initialise global state
    X          = X_nonlin.copy()
    Cset       = [C0, C2]
    Rset       = [R0, R2]
    registered = [0, 1]
    tracks     = [{0: tuple(x1[j]), 1: tuple(x2[j])} for j in range(len(X))]

    # save the very first point cloud
    X_very_beginning = X.copy()

    np.save(os.path.join(OUT, "tracks_init.npy"), tracks)
    np.save(os.path.join(OUT, "X_init.npy"), X)

    # 3) Register cameras 3 to 5
    for i in range(2, num_images):
        print(f"\n[3] Registering camera {i+1}...")

        tracks_for_new_camera(i, tracks, clean)

        X_pnp, x_pnp = get_pnp_correspondences(i, X, tracks)
        print(f"  Found {len(X_pnp)} 3D-2D correspondences for PnP")

        if len(X_pnp) < 6:
            print(f"  Skipping camera {i+1}: not enough correspondences (need ≥6)")
            continue

        C_lin_pnp, R_lin_pnp, pnp_mask = PnPRANSAC(X_pnp, x_pnp, K, epsilon=10.0)
        print(f"  PnP RANSAC inliers: {pnp_mask.sum()} / {len(pnp_mask)}")

        np.save(os.path.join(OUT, f"PnP_linear_cam{i+1}.npy"), np.hstack([C_lin_pnp.reshape(1,3), np.zeros((1,0))]))
        np.save(os.path.join(OUT, f"PnP_inliers_mask_cam{i+1}.npy"), pnp_mask)

        if pnp_mask.sum() < 6:
            print(f"  Skipping camera {i+1}: too few PnP inliers ({pnp_mask.sum()})")
            continue

        C_new, R_new = NonlinearPnP(
            X_pnp[pnp_mask], x_pnp[pnp_mask], K, C_lin_pnp, R_lin_pnp,
            max_nfev=200
        )

        out_pnp = os.path.join(OUT, f"pnp_compare_cam{i+1}.png")
        save_pnp_compare_xz(
            X_pnp,
            C_lin_pnp, R_lin_pnp,
            C_new,     R_new,
            Cset, Rset,
            out_path=out_pnp,
            title=f"Linear vs Nonlinear PnP — camera {i+1}"
        )
        print(f"  Saved {out_pnp}")

        np.save(os.path.join(OUT, f"C_pnp_cam{i+1}.npy"), C_new)
        np.save(os.path.join(OUT, f"R_pnp_cam{i+1}.npy"), R_new)

        Cset.append(C_new)
        Rset.append(R_new)
        registered.append(i)

        ref = find_best_ref(i, registered[:-1], clean)
        ref_idx = registered.index(ref)
        ref_C, ref_R = Cset[ref_idx], Rset[ref_idx]
        print(f"  Using camera {ref+1} as triangulation reference")

        key_fwd = (ref, i)
        key_rev = (i, ref)
        if key_fwd in clean:
            xa, xb = clean[key_fwd]
        elif key_rev in clean:
            xb, xa = clean[key_rev]
        else:
            print(f"  No clean pair found for triangulation, skipping new points")
            xa = xb = None

        if xa is not None:
            X_new_l  = LinearTriangulation(K, ref_C, ref_R, C_new, R_new, xa, xb)
            X_new_nl = NonlinearTriangulation(K, ref_C, ref_R, C_new, R_new, xa, xb, X_new_l)

            np.save(os.path.join(OUT, f"X_new_linear_cam{i+1}.npy"), X_new_l)
            np.save(os.path.join(OUT, f"X_new_nonlinear_cam{i+1}.npy"), X_new_nl)

            X_new_nl = filter_outliers(X_new_nl)
            print(f"  Triangulated {len(X_new_nl)} new points (after filter)")

            out_lin_vs_nl = os.path.join(OUT, f"linear_vs_nonlinear_cam{i+1}.png")
            save_pointcloud_compare_xz(
                X_new_l, X_new_nl,
                out_path=out_lin_vs_nl,
                title=f"Linear vs Nonlinear — camera {i+1}",
                Cset=Cset,
                Rset=Rset
            )
            print(f"  Saved {out_lin_vs_nl}")

            for cam_idx, (C, R, x_obs) in enumerate([(ref_C, ref_R, xa), (C_new, R_new, xb)]):
                real_idx = [ref, i][cam_idx]
                if images[real_idx] is not None:
                    out_reproj = os.path.join(OUT, f"reproj_cam{real_idx+1}_after_cam{i+1}.png")
                    save_reprojection_overlay(
                        images[real_idx], x_obs,
                        reproject(K, C, R, X_new_nl),
                        out_path=out_reproj,
                        title=f"Reprojection cam {real_idx+1} (after cam {i+1})"
                    )
                    print(f"  Saved {out_reproj}")

            for k in range(len(X_new_nl)):
                tracks.append({ref: tuple(xa[k]), i: tuple(xb[k])})
            X = np.vstack([X, X_new_nl])

            np.save(os.path.join(OUT, f"X_after_triangulation_cam{i+1}.npy"), X)

        V = BuildVisibilityMatrix(tracks, num_cameras=len(Cset), num_points=len(X))
        print(f"  Bundle Adjustment ({len(Cset)} cameras, {len(X)} points)...")
        X_before_step = X.copy()
        np.save(os.path.join(OUT, f"X_before_BA_cam{i+1}.npy"), X_before_step)

        Cset, Rset, X = BundleAdjustment(K, Cset, Rset, X, tracks, V, max_nfev=50)

        np.save(os.path.join(OUT, f"X_after_BA_cam{i+1}.npy"), X)
        np.save(os.path.join(OUT, f"Cset_after_BA_cam{i+1}.npy"), np.array(Cset))
        np.save(os.path.join(OUT, f"Rset_after_BA_cam{i+1}.npy"), np.array(Rset))

        out_ba = os.path.join(OUT, f"ba_after_cam{i+1}.png")
        save_ba_before_after_xz(
            X_before_step, X,
            out_path=out_ba,
            title=f"Bundle Adjustment after camera {i+1}",
            Cset=Cset,
            Rset=Rset
        )
        print(f"  Saved {out_ba}")

        X, tracks = filter_tracks_and_points(X, tracks, percentile=90)
        print(f"  After outlier filter — {len(X)} points remaining")
        np.save(os.path.join(OUT, f"X_filtered_after_cam{i+1}.npy"), X)
        np.save(os.path.join(OUT, f"tracks_after_cam{i+1}.npy"), tracks)

    # 4): Final Bundle Adjustment 
    print(f"\n[4] Final Bundle Adjustment ({len(Cset)} cameras, {len(X)} points)...")
    V             = BuildVisibilityMatrix(tracks, num_cameras=len(Cset), num_points=len(X))
    X_before_ba   = X.copy()
    np.save(os.path.join(OUT, "X_before_final_BA.npy"), X_before_ba)

    Cset, Rset, X = BundleAdjustment(K, Cset, Rset, X, tracks, V, max_nfev=50)

    np.save(os.path.join(OUT, "X_after_final_BA.npy"), X)
    np.save(os.path.join(OUT, "Cset_after_final_BA.npy"), np.array(Cset))
    np.save(os.path.join(OUT, "Rset_after_final_BA.npy"), np.array(Rset))

    # 5) Final outputs 
    print("\n[5] Saving final reconstruction...")

    save_final_reconstruction(
        X_before_ba, X, Cset, Rset,
        cam_indices=registered,
        out_path=os.path.join(OUT, "final_reconstruction.png"),
        title="Final Reconstruction (X vs Z)"
    )
    print(f"  Saved {os.path.join(OUT, 'final_reconstruction.png')}")

    out_ba_final = os.path.join(OUT, "ba_final.png")
    save_ba_before_after_xz(
        X_very_beginning, X,
        out_path=out_ba_final,
        title="Initial Bootstrap vs Final Reconstruction",
        Cset=Cset,
        Rset=Rset
    )
    print(f"  Saved {out_ba_final}")

    out_pointcloud = os.path.join(OUT, "final_pointcloud.png")
    save_pointcloud_xz(
        X,
        out_path=out_pointcloud,
        title="Final 3D Point Cloud (X vs Z)"
    )
    print(f"  Saved {out_pointcloud}")

    for cam_idx in range(len(Cset)):
        obs_indices = []
        obs_pixels  = []
        for pt_idx, track in enumerate(tracks):
            if cam_idx in track:
                obs_indices.append(pt_idx)
                obs_pixels.append(track[cam_idx])
        if len(obs_indices) == 0:
            continue

        X_obs = X[obs_indices]
        x_obs = np.array(obs_pixels)

        if images[cam_idx] is not None:
            out_final_reproj = os.path.join(OUT, f"final_reproj_cam{cam_idx+1}.png")
            save_reprojection_overlay(
                images[cam_idx],
                x_obs,
                reproject(K, Cset[cam_idx], Rset[cam_idx], X_obs),
                out_path=out_final_reproj,
                title=f"Final reprojection — camera {cam_idx+1}"
            )
            print(f"  Saved {out_final_reproj}")

    np.save(os.path.join(OUT, "X_final.npy"),    X)
    np.save(os.path.join(OUT, "Cset_final.npy"), np.array(Cset))
    np.save(os.path.join(OUT, "Rset_final.npy"), np.array(Rset))
    np.save(os.path.join(OUT, "tracks_final.npy"), tracks)

    print(f"\nDone. All outputs saved to ./{OUT}/")
    return Cset, Rset, X

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure from Motion")
    parser.add_argument("--data_dir",   default="../P2Data/",                help="Folder with images + matching files")
    parser.add_argument("--calib",      default="../P2Data/calibration.txt", help="Calibration file")
    parser.add_argument("--num_images", default=5, type=int,                 help="Number of images")
    args = parser.parse_args()

    run_sfm(
        data_dir   = args.data_dir,
        calib_file = args.calib,
        num_images = args.num_images
    )