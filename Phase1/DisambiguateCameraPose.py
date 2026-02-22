import numpy as np
from LinearTriangulation import LinearTriangulation


def is_in_front(C: np.ndarray, R: np.ndarray, X: np.ndarray) -> np.ndarray:

    r3 = R[2, :]                   # third row = camera forward axis (3,)
    return (X - C) @ r3 > 0        # (N,) dot product per point


def DisambiguateCameraPose(
    K: np.ndarray,
    Cset: list, Rset: list,
    x1: np.ndarray, x2: np.ndarray
):
    """
    Find the correct camera pose from 4 candidates using cheirality.

    Inputs:
      K:        (3,3) intrinsic matrix
      Cset:     list of 4 candidate camera centers (each (3,))
      Rset:     list of 4 candidate rotation matrices (each (3,3))
      x1, x2:   (N,2) pixel correspondences

    Returns:
      C_best:  (3,)   correct camera center
      R_best:  (3,3)  correct rotation matrix
      X_best:  (N,3)  triangulated 3D points for the best pose
    """
    # Camera 1 is fixed at world origin
    C1 = np.zeros(3)
    R1 = np.eye(3)

    best_i     = -1
    best_count = -1
    best_X     = None

    for i in range(4):
        # Triangulate using this candidate pose
        X = LinearTriangulation(K, C1, R1, Cset[i], Rset[i], x1, x2)  # (N,3)

        # Count points in front of BOTH cameras
        in_front = is_in_front(C1, R1, X) & is_in_front(Cset[i], Rset[i], X)
        count = int(np.sum(in_front))

        if count > best_count:
            best_count = count
            best_i     = i
            best_X     = X

    return Cset[best_i], Rset[best_i], best_X