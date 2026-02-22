import numpy as np
from scipy.optimize import least_squares
from EstimateFundamentalMatrix import make_homogeneous
from LinearTriangulation import camera_projection_matrix


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrix → quaternion (w, x, y, z).
    """
    R = R.astype(np.float64)
    trace = R[0,0] + R[1,1] + R[2,2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s

    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s

    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s

    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) → rotation matrix (3,3)."""
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def project(K, C, R, X) -> np.ndarray:
    """Project X (N,3) → pixels (N,2) through pose (C,R) and intrinsics K."""
    P   = camera_projection_matrix(K, C, R)  # (3,4)
    X_h = make_homogeneous(X)                 # (N,4)
    x_h = (P @ X_h.T).T                       # (N,3)
    return x_h[:, :2] / x_h[:, 2:3]           # (N,2)


def residuals(params, K, X, x_obs):
    """
    Reprojection residuals for all points.
    params = [C(3), q(4)]
    Returns (2N,) residual vector.
    """
    C   = params[:3]
    R   = quat_to_rotmat(params[3:])
    x_hat = project(K, C, R, X)
    return (x_obs - x_hat).ravel()            # (2N,)


def NonlinearPnP(
    X: np.ndarray,
    x: np.ndarray,
    K: np.ndarray,
    C_init: np.ndarray,
    R_init: np.ndarray,
    max_nfev: int = 100
):
    """
    Refine camera pose by minimizing reprojection error via nonlinear least squares.

    Inputs:
      X, x:         (N,3), (N,2) inlier correspondences
      K:            (3,3) intrinsics
      C_init,R_init: initial pose from PnPRANSAC
      max_nfev:     max optimizer iterations

    Returns:
      C_ref: (3,)   refined camera center
      R_ref: (3,3)  refined rotation matrix
    """
    assert X.shape[0] == x.shape[0]
    assert X.shape[0] >= 6

    p0 = np.hstack([C_init.ravel(), rotmat_to_quat(R_init)])

    result = least_squares(
        residuals,
        p0,
        args=(K, X, x),
        method='trf',
        max_nfev=max_nfev
    )

    C_ref = result.x[:3]
    R_ref = quat_to_rotmat(result.x[3:])

    return C_ref, R_ref