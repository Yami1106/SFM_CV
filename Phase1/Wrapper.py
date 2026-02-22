"""
Wrapper.py — Full Structure from Motion pipeline.
All visualizations saved to ./outputs/ via Visualizations.py.

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
    save_ba_before_after_xz,
)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Final reconstruction plot (inlined — not in provided Visualizations.py)
# ─────────────────────────────────────────────────────────────────────────────

def save_final_reconstruction(X_before, X_after, Cset, Rset, out_path, title="Final Reconstruction (X vs Z)"):
    """
    Blue = before BA, red = after BA, green triangles = camera positions.
    Matches the reference image style.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_before[:, 0], X_before[:, 2], s=4, c='blue', alpha=0.5, label='before bund adj')
    ax.scatter(X_after[:,  0], X_after[:,  2], s=4, c='red',  alpha=0.5, label='after bund adj')

    for i, C in enumerate(Cset):
        ax.scatter(C[0], C[2], marker='^', s=120, c='green',
                   zorder=5, edgecolors='black', linewidths=0.5)
        ax.text(C[0] + 0.05, C[2] + 0.05, str(i + 1),
                fontsize=8, color='purple', fontweight='bold', zorder=6)

    ax.legend(fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Outlier filter — returns both filtered array AND boolean keep mask
# ─────────────────────────────────────────────────────────────────────────────

def filter_outliers_with_mask(X: np.ndarray, percentile: int = 95):
    """
    Returns (X_filtered, keep_mask) so callers can also filter tracks in sync.
    """
    if len(X) == 0:
        return X, np.ones(0, dtype=bool)
    median = np.median(X, axis=0)
    dists  = np.linalg.norm(X - median, axis=1)
    thresh = np.percentile(dists, percentile)
    keep   = dists < thresh
    return X[keep], keep


def filter_outliers(X: np.ndarray, percentile: int = 95) -> np.ndarray:
    X_f, _ = filter_outliers_with_mask(X, percentile)
    return X_f


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_intrinsics(calib_path: str) -> np.ndarray:
    return np.loadtxt(calib_path)


def load_image(data_dir: str, idx: int):
    for ext in (".jpg", ".png", ".JPG", ".PNG"):
        p = os.path.join(data_dir, f"{idx+1}{ext}")
        if os.path.exists(p):
            return cv2.imread(p)
    return None


def load_matches(data_dir: str):
    """
    Format per line:
      nMatches R G B u1 v1  target u2 v2  [target u2 v2 ...]
    Returns:
      matches: dict (i,j) -> (x_i (N,2), x_j (N,2))  0-based indices
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def reproject(K, C, R, X) -> np.ndarray:
    P   = camera_projection_matrix(K, C, R)
    X_h = make_homogeneous(X)
    x_h = (P @ X_h.T).T
    return x_h[:, :2] / x_h[:, 2:3]


def collect_pnp_correspondences(cam_i, X, tracks):
    X_list, x_list = [], []
    for j, track in enumerate(tracks):
        if cam_i in track:
            if j >= len(X):
                # tracks and X are out of sync — skip stale track
                continue
            X_list.append(X[j])
            x_list.append(track[cam_i])
    if not X_list:
        return np.zeros((0, 3)), np.zeros((0, 2))
    return np.array(X_list), np.array(x_list)


def populate_tracks_for_new_camera(cam_i, tracks, clean):
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


def find_best_reference_camera(cam_i, registered, clean):
    """Find already-registered camera with most matches to cam_i."""
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


def filter_tracks_and_points(X: np.ndarray, tracks: list, percentile: int = 90):
    """
    Filter outlier 3D points and remove the corresponding track entries,
    keeping X and tracks in sync.

    Returns:
      X_filtered:      (M,3) filtered points
      tracks_filtered: list of M track dicts
    """
    X_filtered, keep = filter_outliers_with_mask(X, percentile)
    tracks_filtered  = [tracks[j] for j in range(len(tracks)) if keep[j]]
    return X_filtered, tracks_filtered


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_sfm(data_dir: str, calib_file: str, num_images: int):

    print("=" * 60)
    print("Structure from Motion — Full Pipeline")
    print("=" * 60)

    K       = load_intrinsics(calib_file)
    matches = load_matches(data_dir)
    images  = [load_image(data_dir, i) for i in range(num_images)]

    # ── Step 1: RANSAC filter all pairs ──────────────────────────
    print("\n[1] Filtering correspondences with RANSAC...")
    clean = {}
    for (i, j), (x_i, x_j) in matches.items():
        mask         = GetInliersRANSAC(x_i, x_j)
        clean[(i,j)] = (x_i[mask], x_j[mask])
        print(f"  Pair ({i+1},{j+1}): {mask.sum()} / {len(mask)} inliers")

        if images[i] is not None and images[j] is not None:
            save_matches_ransac(
                images[i], images[j], x_i, x_j, mask,
                out_path=os.path.join(OUT, f"matches_{i+1}_{j+1}.png")
            )

    # ── Step 2: Bootstrap — images 1 & 2 ─────────────────────────
    print("\n[2] Bootstrapping from images 1 & 2...")
    x1, x2 = clean[(0, 1)]

    F  = EstimateFundamentalMatrix(x1, x2)
    E  = EfromF(F, K)
    Cset_cand, Rset_cand = ExtractCameraPose(E)

    C0, R0    = np.zeros(3), np.eye(3)
    Xset_cand = [
        LinearTriangulation(K, C0, R0, Cset_cand[k], Rset_cand[k], x1, x2)
        for k in range(4)
    ]

    for k in range(4):
        save_pointcloud_xz(
            Xset_cand[k],
            out_path=os.path.join(OUT, f"candidate_{k+1}_xz.png"),
            title=f"Candidate pose {k+1}"
        )

    C2, R2, X_lin = DisambiguateCameraPose(K, Cset_cand, Rset_cand, x1, x2)
    print(f"  Pose disambiguated — {len(X_lin)} points triangulated")

    X_nonlin = NonlinearTriangulation(K, C0, R0, C2, R2, x1, x2, X_lin)
    X_nonlin = filter_outliers(X_nonlin)
    print(f"  After outlier filter — {len(X_nonlin)} points remaining")

    save_pointcloud_compare_xz(
        X_lin, X_nonlin,
        out_path=os.path.join(OUT, "linear_vs_nonlinear_1_2.png"),
        title="Linear vs Nonlinear — images 1 & 2"
    )

    for cam_idx, (C, R, x_obs) in enumerate([(C0, R0, x1), (C2, R2, x2)]):
        if images[cam_idx] is not None:
            save_reprojection_overlay(
                images[cam_idx], x_obs,
                reproject(K, C, R, X_lin),
                out_path=os.path.join(OUT, f"reproj_linear_cam{cam_idx+1}.png"),
                title=f"Linear reprojection — camera {cam_idx+1}"
            )
            save_reprojection_overlay(
                images[cam_idx], x_obs,
                reproject(K, C, R, X_nonlin),
                out_path=os.path.join(OUT, f"reproj_nonlinear_cam{cam_idx+1}.png"),
                title=f"Nonlinear reprojection — camera {cam_idx+1}"
            )

    # initialise global state
    X          = X_nonlin.copy()
    Cset       = [C0, C2]
    Rset       = [R0, R2]
    registered = [0, 1]
    tracks     = [{0: tuple(x1[j]), 1: tuple(x2[j])} for j in range(len(X))]

    # ── Step 3: Register cameras 3, 4, 5 incrementally ───────────
    for i in range(2, num_images):
        print(f"\n[3] Registering camera {i+1}...")

        populate_tracks_for_new_camera(i, tracks, clean)

        X_pnp, x_pnp = collect_pnp_correspondences(i, X, tracks)
        print(f"  Found {len(X_pnp)} 3D-2D correspondences for PnP")

        if len(X_pnp) < 6:
            print(f"  Skipping camera {i+1}: not enough correspondences (need ≥6)")
            continue

        C_new, R_new, pnp_mask = PnPRANSAC(X_pnp, x_pnp, K, epsilon=10.0)
        print(f"  PnP RANSAC inliers: {pnp_mask.sum()} / {len(pnp_mask)}")

        if pnp_mask.sum() < 6:
            print(f"  Skipping camera {i+1}: too few PnP inliers ({pnp_mask.sum()})")
            continue

        C_new, R_new = NonlinearPnP(
            X_pnp[pnp_mask], x_pnp[pnp_mask], K, C_new, R_new,
            max_nfev=200
        )

        Cset.append(C_new)
        Rset.append(R_new)
        registered.append(i)

        # use best reference camera for triangulation (reduces drift)
        ref       = find_best_reference_camera(i, registered[:-1], clean)
        ref_idx   = registered.index(ref)
        ref_C     = Cset[ref_idx]
        ref_R     = Rset[ref_idx]
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
            X_new_nl = filter_outliers(X_new_nl)
            print(f"  Triangulated {len(X_new_nl)} new points (after filter)")

            save_pointcloud_compare_xz(
                X_new_l, X_new_nl,
                out_path=os.path.join(OUT, f"linear_vs_nonlinear_cam{i+1}.png"),
                title=f"Linear vs Nonlinear — camera {i+1}"
            )

            for cam_idx, (C, R, x_obs) in enumerate([(ref_C, ref_R, xa), (C_new, R_new, xb)]):
                real_idx = [ref, i][cam_idx]
                if images[real_idx] is not None:
                    save_reprojection_overlay(
                        images[real_idx], x_obs,
                        reproject(K, C, R, X_new_nl),
                        out_path=os.path.join(OUT, f"reproj_cam{real_idx+1}_after_cam{i+1}.png"),
                        title=f"Reprojection cam {real_idx+1} (after cam {i+1})"
                    )

            # Append new points and their tracks together, then sync-filter
            for k in range(len(X_new_nl)):
                tracks.append({ref: tuple(xa[k]), i: tuple(xb[k])})
            X = np.vstack([X, X_new_nl])

        # BA after each camera
        V = BuildVisibilityMatrix(tracks, num_cameras=len(Cset), num_points=len(X))
        print(f"  Bundle Adjustment ({len(Cset)} cameras, {len(X)} points)...")
        Cset, Rset, X = BundleAdjustment(K, Cset, Rset, X, tracks, V, max_nfev=50)

        # ── FIX: filter outliers from BOTH X and tracks in sync ──
        X, tracks = filter_tracks_and_points(X, tracks, percentile=90)
        print(f"  After outlier filter — {len(X)} points remaining")

    # ── Step 4: Final BA ──────────────────────────────────────────
    print(f"\n[4] Final Bundle Adjustment ({len(Cset)} cameras, {len(X)} points)...")
    V             = BuildVisibilityMatrix(tracks, num_cameras=len(Cset), num_points=len(X))
    X_before_ba   = X.copy()
    Cset, Rset, X = BundleAdjustment(K, Cset, Rset, X, tracks, V, max_nfev=50)

    # ── Step 5: Final outputs ─────────────────────────────────────
    print("\n[5] Saving final reconstruction...")

    save_final_reconstruction(
        X_before_ba, X, Cset, Rset,
        out_path=os.path.join(OUT, "final_reconstruction.png"),
        title="Final Reconstruction (X vs Z)"
    )

    save_ba_before_after_xz(
        X_before_ba, X,
        out_path=os.path.join(OUT, "ba_final.png"),
        title="Final Bundle Adjustment"
    )

    save_pointcloud_xz(
        X,
        out_path=os.path.join(OUT, "final_pointcloud.png"),
        title="Final 3D Point Cloud (X vs Z)"
    )

    np.save(os.path.join(OUT, "X_final.npy"),    X)
    np.save(os.path.join(OUT, "Cset_final.npy"), np.array(Cset))
    np.save(os.path.join(OUT, "Rset_final.npy"), np.array(Rset))

    print(f"\nDone. All outputs saved to ./{OUT}/")
    return Cset, Rset, X


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure from Motion")
    parser.add_argument("--data_dir",   default="../P2Data/",                help="Folder with images + matching files")
    parser.add_argument("--calib",      default="../P2Data/calibration.txt", help="Calibration file")
    parser.add_argument("--num_images", default=5, type=int,            help="Number of images")
    args = parser.parse_args()

    run_sfm(
        data_dir   = args.data_dir,
        calib_file = args.calib,
        num_images = args.num_images
    )