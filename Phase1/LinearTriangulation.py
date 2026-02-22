import numpy as np
from EstimateFundamentalMatrix import make_homogeneous


def camera_projection_matrix(K: np.ndarray, C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """P = K R [I | -C]"""
    t = -R @ C.reshape(3, 1)
    return K @ np.hstack([R, t])  # (3,4)


def triangulate_point(P1, P2, x1, x2) -> np.ndarray:
    """
    Builds 4x4 system AX=0, solve using SVD.
    """
    u1, v1 = x1[0], x1[1]
    u2, v2 = x2[0], x2[1]

    A = np.array([
        u1 * P1[2, :] - P1[0, :],   # constraint from x1
        v1 * P1[2, :] - P1[1, :],   # constraint from y1
        u2 * P2[2, :] - P2[0, :],   # constraint from x2
        v2 * P2[2, :] - P2[1, :]    # constraint from y2
    ], dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]              
    return X_h[:3] / X_h[3] 


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2) -> np.ndarray:
    """
    Triangulate N correspondences.

    Inputs:
      K:      (3,3) intrinsic matrix
      C1, C2: (3,)  camera centers in world coords
      R1, R2: (3,3) rotation matrices
      x1, x2: (N,2) pixel correspondences

    Returns:
      X: (N,3) triangulated 3D points
    """
    assert x1.shape == x2.shape

    P1 = camera_projection_matrix(K, C1, R1)
    P2 = camera_projection_matrix(K, C2, R2)

    x1h = make_homogeneous(x1)  # (N,3)
    x2h = make_homogeneous(x2)  # (N,3)

    N = x1.shape[0]
    X = np.zeros((N, 3))

    for i in range(N):
        X[i] = triangulate_point(P1, P2, x1h[i], x2h[i])

    return X