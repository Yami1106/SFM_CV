# Phase1/EstimateFundamentalMatrix.py
import numpy as np

def make_homogeneous(x: np.ndarray) -> np.ndarray:
    """
    x: (N,2)
    returns: (N,3) homogeneous points

    convert (x,y) to (x,y,1)
    """

    ones = np.ones((x.shape[0], 1))
    x_h = np.hstack([x, ones])

    return x_h

def normalize_points_2d(x: np.ndarray):
    """
    Args:
      x: (N,2) pixel coordinates

    Returns:
      x_norm: (N,2)
      T: (3,3) normalization transform such that x_h_norm ~ T @ x_h
    """

    x_homogeneous = make_homogeneous(x) 
    centroid = np.mean(x, axis=0)
    Tx = -centroid[0]
    Ty = -centroid[1]
    T_translate = np.array([[1, 0, Tx],
                            [0, 1, Ty],
                            [0, 0, 1]])
    
    translated_points = (T_translate @ x_homogeneous.T).T

    avg_distance = np.mean(np.sqrt(np.sum(translated_points[:, :2] ** 2, axis=1)))

    scale = np.sqrt(2) / avg_distance
    T_scale = np.array([[scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, 1]]) 
    
    T = T_scale @ T_translate
    normalized_points = (T @ x_homogeneous.T).T

    return normalized_points, T

def build_A(x1n: np.ndarray, x2n: np.ndarray) -> np.ndarray:
    """
    Build the A matrix for normalized correspondences.

    x1n, x2n: (N,2) normalized coordinates for image1 and image2
    A: (N,9) each row corresponds to:
      [x2*x1, x2*y1, x2,
       y2*x1, y2*y1, y2,
       x1,    y1,    1]
    (Depending on your convention, verify the row ordering carefully.)

    Returns:
      A: (N,9)
    """
    
    N = x1n.shape[0]
    A = np.zeros((N, 9))

    for i in range(N):
        x1, y1 = x1n[i]
        x2, y2 = x2n[i]

        A[i] = [x2 * x1, x2 * y1, x2,
                y2 * x1, y2 * y1, y2,
                x1,      y1,      1]

    return A

def get_F_from_A(A: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1]         
    F = f.reshape(3, 3) 
    return F

def force_rank2(F: np.ndarray) -> np.ndarray:
    """
    Enforce rank-2 constraint on F by SVD and zeroing smallest singular value.
    """

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  
    F_rank2 = U @ np.diag(S) @ Vt
    return F_rank2

def EstimateFundamentalMatrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Estimate Fundamental Matrix F such that x2^T F x1 = 0.

    Args:
      x1: (N,2) points in image 1
      x2: (N,2) corresponding points in image 2
          N must be >= 8

    Returns:
      F: (3,3) fundamental matrix 
    """

    # make sure the number of points from both images is greater than or equal to 8 and they match
    assert x1.shape == x2.shape
    assert x1.shape[0] >= 8

    # 1) Normalize points
    x1n, T1 = normalize_points_2d(x1)
    x2n, T2 = normalize_points_2d(x2)

    # 2) Build A for the linear system Af = 0
    # build A expects (N,2) normalized coordinates and not (N,3) hence slice it before passing to it
    A = build_A(x1n[:, :2], x2n[:, :2])

    # 3) Solve Af = 0 with SVD
    F_norm = get_F_from_A(A)

    # 4) Enforce rank-2 on F_norm
    F_rank2 = force_rank2(F_norm)

    # 5) Denormalize: F = T2^T * F_rank2 * T1
    F = T2.T @ F_rank2 @ T1

    return F

def epipolar_residuals(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Get epipolar residuals: r_i = x2_i^T F x1_i
    
    Args:
      F:  (3,3) fundamental matrix
      x1: (N,2) points in image 1
      x2: (N,2) points in image 2
    
    Returns:
      residuals: (N,) one scalar for eveery correspondence, should be close to 0 for good matches
    """

    # make homogeneous (N,3)
    x1h = make_homogeneous(x1)
    x2h = make_homogeneous(x2)

    # for each i: x2[i]^T @ F @ x1[i]
    # F @ x1h.T gives (3,N), then multiply elementwise with x2h.T and sum rows
    Fx1 = F @ x1h.T                        # (3,N)
    residuals = np.sum(x2h.T * Fx1, axis=0) # (N,)

    return residuals
    
    