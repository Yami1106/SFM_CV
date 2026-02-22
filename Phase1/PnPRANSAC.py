import numpy as np
from LinearPnP import LinearPnP
from LinearTriangulation import camera_projection_matrix
from EstimateFundamentalMatrix import make_homogeneous


def project(K: np.ndarray, C: np.ndarray, R: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Project 3D points X (N,3) → pixel coords (N,2).
    Builds P = K[R|t] then divides by z.
    """
    P   = camera_projection_matrix(K, C, R)  # (3,4)
    X_h = make_homogeneous(X)                 # (N,4)
    x_h = (P @ X_h.T).T                       # (N,3)
    return x_h[:, :2] / x_h[:, 2:3]           # (N,2)


def reprojection_error(K, C, R, X, x_obs) -> np.ndarray:
    """
    Per-point L2 reprojection error in pixels.
    Returns (N,) array.
    """
    x_hat = project(K, C, R, X)              # (N,2)
    return np.linalg.norm(x_obs - x_hat, axis=1)  # (N,)


def PnPRANSAC(
    X: np.ndarray,
    x: np.ndarray,
    K: np.ndarray,
    M: int   = 2000,
    epsilon: float = 2.0,
    seed: int = 0
):
    """
    camera pose estimation via RANSAC + LinearPnP.

    Inputs:
      X:       (N,3) 3D world points
      x:       (N,2) corresponding 2D pixel points
      K:       (3,3) intrinsic matrix
      M:       number of RANSAC iterations
      epsilon: inlier threshold in pixels (L2 reprojection error)

    Returns:
      C_best:      (3,)   best camera center
      R_best:      (3,3)  best rotation matrix
      best_inliers:(N,)   boolean mask of inliers
    """
    assert X.shape[0] == x.shape[0]
    N = X.shape[0]
    assert N >= 6

    rng = np.random.default_rng(seed)

    best_count   = -1
    best_inliers = np.zeros(N, dtype=bool)
    C_best       = None
    R_best       = None

    for _ in range(M):
        # 1) Sample 6 random correspondences
        idx = rng.choice(N, size=6, replace=False)

        # 2) Estimate pose from 6 points
        try:
            C, R = LinearPnP(X[idx], x[idx], K)
        except Exception:
            continue   

        # 3) Reprojection error for ALL N points
        err = reprojection_error(K, C, R, X, x)  # (N,)

        # 4) Count inliers
        inliers = err < epsilon
        count   = int(np.sum(inliers))

        # 5) Keep best pose
        if count > best_count:
            best_count   = count
            best_inliers = inliers
            C_best, R_best = C, R

    if C_best is None:
        C_best, R_best = LinearPnP(X, x, K)
        best_inliers   = np.ones(N, dtype=bool)

    return C_best, R_best, best_inliers