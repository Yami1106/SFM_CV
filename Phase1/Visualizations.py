# Phase1/Visualizations.py
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt


def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)


def _save_fig(path: str, dpi: int = 200):
    ensure_dir(str(Path(path).parent))
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def save_matches_ransac(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    inlier_mask: np.ndarray,
    out_path: str,
    max_draw: int = 2000
):
    """
    Side-by-side match visualization.
    green lines = inliers,  red lines = outliers
    """
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1]      = img1
    canvas[:h2, w1:w1+w2] = img2

    x1 = x1.astype(np.int32)
    x2 = x2.astype(np.int32)

    idx = np.arange(x1.shape[0])
    if x1.shape[0] > max_draw:
        idx = np.random.choice(idx, size=max_draw, replace=False)

    for k in idx:
        p1  = (int(x1[k, 0]),      int(x1[k, 1]))
        p2  = (int(x2[k, 0] + w1), int(x2[k, 1]))
        col = (0, 255, 0) if bool(inlier_mask[k]) else (255, 0, 0)
        cv2.line(canvas, p1, p2, col, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, col, -1)
        cv2.circle(canvas, p2, 2, col, -1)

    ensure_dir(str(Path(out_path).parent))
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  Saved {out_path}")

def save_reprojection_overlay(
    img_bgr: np.ndarray,
    x_obs: np.ndarray,
    x_proj: np.ndarray,
    out_path: str,
    title: str = ""
):
    """
    Single image reprojection overlay.
    green = observed features,  red = reprojected points
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.scatter(x_obs[:,  0], x_obs[:,  1], s=6, c='lime',  label='detected')
    plt.scatter(x_proj[:, 0], x_proj[:, 1], s=6, c='red',   label='reprojected')
    plt.legend(fontsize=7)
    plt.axis('off')
    _save_fig(out_path)

def save_pointcloud_xz(
    X: np.ndarray,
    out_path: str,
    title: str = "Point Cloud (X vs Z)",
    labels=None,
    s: int = 4
):
    """
    X vs Z scatter.  If labels provided, colours each group separately.
    """
    X = np.asarray(X)
    plt.figure(figsize=(6, 6))
    if labels is None:
        plt.scatter(X[:, 0], X[:, 2], s=s)
    else:
        labels = np.asarray(labels)
        for lab in np.unique(labels):
            pts = X[labels == lab]
            plt.scatter(pts[:, 0], pts[:, 2], s=s, label=str(lab))
        plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("z")
    _save_fig(out_path)


def save_pointcloud_compare_xz(
    X_a: np.ndarray,
    X_b: np.ndarray,
    out_path: str,
    label_a: str = "linear",
    label_b: str = "nonlinear",
    title: str = "Linear vs Nonlinear (X vs Z)",
    s: int = 6
):
    """
    Overlay comparison — red = linear, blue = nonlinear (matches your screenshots).
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(X_a[:, 0], X_a[:, 2], s=s, c='red',  alpha=0.7, label=label_a)
    plt.scatter(X_b[:, 0], X_b[:, 2], s=s, c='blue', alpha=0.7, label=label_b)
    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("z")
    _save_fig(out_path)

def save_ba_before_after_xz(
    X_before: np.ndarray,
    X_after: np.ndarray,
    out_path: str,
    title: str = "Before vs After Bundle Adjustment (X vs Z)",
    s: int = 5
):
    """
    BA comparison — blue = before, red = after.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(X_before[:, 0], X_before[:, 2], s=s, c='blue', alpha=0.6, label='before bund adj')
    plt.scatter(X_after[:,  0], X_after[:,  2], s=s, c='red',  alpha=0.6, label='after bund adj')
    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("z")
    _save_fig(out_path)