# Phase1/Visualizations.py
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D


def check_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)


def save_img(path: str, dpi: int = 200):
    check_dir(str(Path(path).parent))
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

def camera_frustum(ax, C, R, color='green', label=None, size=0.4):
    forward  = R.T[:, 2]
    right    = R.T[:, 0]

    tip      = np.array([C[0], C[2]])
    left     = np.array([C[0] - right[0] * size * 0.7 + forward[0] * size * 0.1,
                         C[2] - right[2] * size * 0.7 + forward[2] * size * 0.1])
    right_pt = np.array([C[0] + right[0] * size * 0.7 + forward[0] * size * 0.1,
                         C[2] + right[2] * size * 0.7 + forward[2] * size * 0.1])

    triangle = plt.Polygon(
        [tip, left, right_pt],
        closed=True,
        facecolor=color,
        edgecolor='black',
        linewidth=0.5,
        zorder=5,
        label=label
    )
    ax.add_patch(triangle)


def camera_colors(n):
    base = ['#6B2D8B', '#D2691E', '#B8860B', '#2E8B57', '#DC143C',
            '#4169E1', '#FF8C00', '#008B8B', '#8B008B', '#556B2F']
    return [base[i % len(base)] for i in range(n)]

def save_matches_ransac(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    inlier_mask: np.ndarray,
    out_path: str,
    max_draw: int = 2000
):
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

    check_dir(str(Path(out_path).parent))
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  Saved {out_path}")


def save_reprojection_overlay(
    img_bgr: np.ndarray,
    x_obs: np.ndarray,
    x_proj: np.ndarray,
    out_path: str,
    title: str = ""
):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(img)
    if title:
        ax.set_title(title, fontsize=10)
    ax.scatter(x_obs[:,  0], x_obs[:,  1], s=8, c='lime',  label='detected',    zorder=3)
    ax.scatter(x_proj[:, 0], x_proj[:, 1], s=8, c='red',   label='reprojected', zorder=4)
    ax.legend(fontsize=8, loc='upper right')
    save_img(out_path)


def save_candidate_poses_xz(
    Xset: list,
    out_path: str,
    title: str = "initial triangulation"
):
    colors = ['#8B4513', '#FF69B4', '#6A0DAD', '#1E90FF']

    fig, ax = plt.subplots(figsize=(6, 7))
    for k, X in enumerate(Xset):
        X = np.asarray(X)
        if len(X) == 0:
            continue
        ax.scatter(X[:, 0], X[:, 2], s=3, c=colors[k % len(colors)], alpha=0.7)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    save_img(out_path)


def save_pointcloud_xz(
    X: np.ndarray,
    out_path: str,
    title: str = "Point Cloud (X vs Z)",
    labels=None,
    s: int = 4
):
    X = np.asarray(X)
    fig, ax = plt.subplots(figsize=(6, 6))
    if labels is None:
        ax.scatter(X[:, 0], X[:, 2], s=s)
    else:
        labels = np.asarray(labels)
        palette = ['#1E90FF', '#FF4500', '#2ECC71', '#9B59B6', '#E67E22']
        for ki, lab in enumerate(np.unique(labels)):
            pts = X[labels == lab]
            ax.scatter(pts[:, 0], pts[:, 2], s=s,
                       c=palette[ki % len(palette)], label=str(lab))
        ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    save_img(out_path)


def save_pointcloud_compare_xz(
    X_a: np.ndarray,
    X_b: np.ndarray,
    out_path: str,
    label_a: str = "linear",
    label_b: str = "nonlinear",
    title: str = "Linear vs Nonlinear (X vs Z)",
    s: int = 6,
    Cset=None,
    Rset=None
):
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.scatter(np.asarray(X_a)[:, 0], np.asarray(X_a)[:, 2],
               s=s, c='red',  alpha=0.7, label=label_a)
    ax.scatter(np.asarray(X_b)[:, 0], np.asarray(X_b)[:, 2],
               s=s, c='blue', alpha=0.7, label=label_b)

    if Cset is not None and Rset is not None:
        cam_colors = camera_colors(len(Cset))
        for ci, (C, R) in enumerate(zip(Cset, Rset)):
            camera_frustum(ax, C, R, color=cam_colors[ci], size=0.35)

    ax.legend(fontsize=8, loc='upper left')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    save_img(out_path)


def save_pnp_compare_xz(
    X: np.ndarray,
    C_lin: np.ndarray,
    R_lin: np.ndarray,
    C_nonlin: np.ndarray,
    R_nonlin: np.ndarray,
    Cset_prev: list,
    Rset_prev: list,
    out_path: str,
    title: str = "Linear vs Nonlinear PnP",
    s: int = 5
):
    fig, ax = plt.subplots(figsize=(5, 6))

    X = np.asarray(X)
    if len(X) > 0:
        ax.scatter(X[:, 0], X[:, 2],
                   s=3, c='lightgray', alpha=0.6, zorder=1, label='3D points')

    C_lin = np.asarray(C_lin)
    ax.scatter(C_lin[0], C_lin[2],
               s=18, c='red', zorder=5, label='linear')

    C_nonlin = np.asarray(C_nonlin)
    ax.scatter(C_nonlin[0], C_nonlin[2],
               s=18, c='blue', zorder=5, label='nonlinear')

    ax.legend(fontsize=8, loc='upper right',
              frameon=True, framealpha=1.0, edgecolor='black')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    save_img(out_path)


def save_ba_before_after_xz(
    X_before: np.ndarray,
    X_after: np.ndarray,
    out_path: str,
    title: str = "Before vs After Bundle Adjustment (X vs Z)",
    Cset=None,
    Rset=None,
    cam_indices: list = None,
    s: int = 5,
    xlim=(-15, 15),
    zlim=(-5, 25)
):
    fig, ax = plt.subplots(figsize=(7, 8))

    ax.scatter(np.asarray(X_before)[:, 0], np.asarray(X_before)[:, 2],
               s=s, c='blue', alpha=0.5, label='before bund adj', zorder=2)
    ax.scatter(np.asarray(X_after)[:,  0], np.asarray(X_after)[:,  2],
               s=s, c='red',  alpha=0.5, label='after bund adj',  zorder=2)

    if Cset is not None and Rset is not None:
        cam_colors = camera_colors(len(Cset))
        for ci, (C, R) in enumerate(zip(Cset, Rset)):
            label = str(cam_indices[ci] + 1) if cam_indices is not None else str(ci + 1)
            camera_frustum(ax, C, R, color=cam_colors[ci], size=2.5)
            ax.text(C[0], C[2] + 1.2, label,
                    fontsize=14, color='white', fontweight='bold',
                    ha='center', va='center', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc=cam_colors[ci],
                              ec='black', lw=1.5))

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend(fontsize=9, loc='lower left',
              handles=[mpatches.Patch(color='blue', label='before bund adj'),
                       mpatches.Patch(color='red',  label='after bund adj')])
    save_img(out_path)


def save_final_reconstruction(
    X_before: np.ndarray,
    X_after: np.ndarray,
    Cset: list,
    Rset: list,
    out_path: str,
    cam_indices: list = None,
    title: str = "Final Reconstruction (X vs Z)",
    xlim=(-15, 15),
    zlim=(-5, 25)
):
    fig, ax = plt.subplots(figsize=(7, 8))

    ax.scatter(np.asarray(X_before)[:, 0], np.asarray(X_before)[:, 2],
               s=4, c='blue', alpha=0.45, zorder=2, label='before bund adj')
    ax.scatter(np.asarray(X_after)[:,  0], np.asarray(X_after)[:,  2],
               s=4, c='red',  alpha=0.45, zorder=2, label='after bund adj')

    cam_colors = camera_colors(len(Cset))
    for ci, (C, R) in enumerate(zip(Cset, Rset)):
        label = str(cam_indices[ci] + 1) if cam_indices is not None else str(ci + 1)
        camera_frustum(ax, C, R, color=cam_colors[ci], size=2.5)
        ax.text(C[0], C[2] + 1.2, label,
                fontsize=14, color='white', fontweight='bold',
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', fc=cam_colors[ci],
                          ec='black', lw=1.5))

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend(fontsize=9, loc='lower left',
              handles=[mpatches.Patch(color='blue', label='before bund adj'),
                       mpatches.Patch(color='red',  label='after bund adj')])
    check_dir(str(Path(out_path).parent))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()