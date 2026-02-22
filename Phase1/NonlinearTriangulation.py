import numpy as np
from scipy.optimize import least_squares
from LinearTriangulation import camera_projection_matrix


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Project 3D point X (3,) through P (3,4) → pixel (u,v).
    """
    X_h = np.append(X, 1.0) 
    x_h = P @ X_h              
    return x_h[:2] / x_h[2]   


def residuals(X, P1, P2, x1, x2):
    """
    4-vector of reprojection residuals for one 3D point across both cameras.
    least_squares minimizes sum of squares of this vector.
    """
    x1_hat = project(P1, X)
    x2_hat = project(P2, X)
    return np.array([
        x1[0] - x1_hat[0],   # u error in camera 1
        x1[1] - x1_hat[1],   # v error in camera 1
        x2[0] - x2_hat[0],   # u error in camera 2
        x2[1] - x2_hat[1],   # v error in camera 2
    ])


def NonlinearTriangulation(
    K: np.ndarray,
    C1: np.ndarray, R1: np.ndarray,
    C2: np.ndarray, R2: np.ndarray,
    x1: np.ndarray, x2: np.ndarray,
    X_init: np.ndarray
) -> np.ndarray:
    """
    Refine 3D points by minimizing reprojection error via nonlinear least squares.

    Inputs:
      K:        (3,3) intrinsic matrix
      C1,R1:    camera 1 pose
      C2,R2:    camera 2 pose
      x1,x2:    (N,2) pixel correspondences
      X_init:   (N,3) initial guess from LinearTriangulation

    Returns:
      X_refined: (N,3) geometrically optimal 3D points
    """
    assert x1.shape == x2.shape
    assert X_init.shape[0] == x1.shape[0]

    P1 = camera_projection_matrix(K, C1, R1)
    P2 = camera_projection_matrix(K, C2, R2)

    N = x1.shape[0]
    X_refined = np.zeros((N, 3))

    for i in range(N):
        result = least_squares(
            residuals,
            X_init[i],
            args=(P1, P2, x1[i], x2[i]), 
            method='trf',
            max_nfev=50
        )
        X_refined[i] = result.x

    return X_refined