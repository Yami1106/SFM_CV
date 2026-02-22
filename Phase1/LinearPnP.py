import numpy as np
from EstimateFundamentalMatrix import make_homogeneous


def normalize_points(K: np.ndarray, x: np.ndarray) -> np.ndarray:
    """normalized camera coords: x_n = K⁻¹ x, returns (N,3)."""
    x_h = make_homogeneous(x)              # (N,3)
    return (np.linalg.inv(K) @ x_h.T).T   # (N,3)


def pnp_dlt(X: np.ndarray, x_n: np.ndarray) -> np.ndarray:
    """
    solve for P_n = [R|t] from 3D-2D correspondences.
    Builds A (2N x 12), solves Ap=0 via SVD.
    Returns P_n (3,4).
    """
    N = X.shape[0]
    X_h = make_homogeneous(X)              # (N,4)

    u = x_n[:, 0] / x_n[:, 2]             # (N,)
    v = x_n[:, 1] / x_n[:, 2]             # (N,)

    A = np.zeros((2 * N, 12))

    for i in range(N):
        Xi = X_h[i]
        A[2*i,   0:4]  =  Xi
        A[2*i,   8:12] = -u[i] * Xi
        A[2*i+1, 4:8]  =  Xi
        A[2*i+1, 8:12] = -v[i] * Xi

    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 4)


def get_R_t(P_n: np.ndarray):
    """
    Extract R ∈ SO(3) and t from P_n ≈ [R|t].
    Enforces orthonormality through SVD and flips if det(R) = -1.
    """
    M  = P_n[:, :3]
    p4 = P_n[:,  3]

    scale    = 1.0 / np.linalg.norm(M[:, 0])
    U, _, Vt = np.linalg.svd(scale * M)
    R        = U @ Vt
    t        = scale * p4

    if np.linalg.det(R) < 0:
        R, t = -R, -t

    return R, t


def LinearPnP(X: np.ndarray, x: np.ndarray, K: np.ndarray):
    """
    Estimate camera pose from 2D-3D correspondences via DLT.

    Inputs:
      X: (N,3) 3D world points
      x: (N,2) 2D pixel points
      K: (3,3) intrinsic matrix

    Returns:
      C: (3,)   camera center in world coords
      R: (3,3)  rotation matrix
    """
    assert X.shape[0] == x.shape[0]
    assert X.shape[0] >= 6

    x_n     = normalize_points(K, x)   # remove K from the problem
    P_n     = pnp_dlt(X, x_n)          # solve for [R|t]
    R, t    = get_R_t(P_n)             # force R ∈ SO(3)
    C       = -R.T @ t                 # t = -RC → C = -Rᵀt

    return C, R