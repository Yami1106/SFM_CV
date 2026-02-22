import numpy as np
from scipy.optimize import least_squares
from NonlinearPnP import rotmat_to_quat, quat_to_rotmat
from LinearTriangulation import camera_projection_matrix
from EstimateFundamentalMatrix import make_homogeneous


def project_single(K, C, q, X):
    """
    Project one 3D point X (3,) through camera (C, q) → (u, v).
    """
    R   = quat_to_rotmat(q)
    P   = camera_projection_matrix(K, C, R)      # (3,4)
    X_h = np.append(X, 1.0)                        # (4,)
    x_h = P @ X_h                                  # (3,)
    return x_h[0] / x_h[2], x_h[1] / x_h[2]


def pack(Cset, Rset, X):
    """
    Pack all unknowns into flat 1D vector for least_squares.
    Total:  7I(3 for C and 4 for q) + 3J (3 for X per point)
    """
    params = []
    for C, R in zip(Cset, Rset):
        params.append(C.ravel())
        params.append(rotmat_to_quat(R))
    params.append(X.ravel())
    return np.concatenate(params)


def unpack(params, I, J):
    """
    Unpack vector → Cset, qset, X.
    """
    idx  = 0
    Cset = []
    qset = []

    for _ in range(I):
        Cset.append(params[idx:idx+3]); idx += 3
        qset.append(params[idx:idx+4]); idx += 4

    X = params[idx:idx+3*J].reshape(J, 3)
    return Cset, qset, X


def residuals(params, K, V, tracks, I, J):
    """
    Reprojection residuals for all visible (camera, point) pairs.

    Only computes residuals where V[i,j] = 1.
    Returns 1D vector of length 2 * sum(V).
    """
    Cset, qset, X = unpack(params, I, J)
    res = []

    for i in range(I):
        R = quat_to_rotmat(qset[i])
        P = camera_projection_matrix(K, Cset[i], R)  # (3,4)

        # get all j where V[i,j] = 1
        visible_js = np.where(V[i] == 1)[0]
        if len(visible_js) == 0:
            continue

        # project all visible points at once
        X_vis = X[visible_js]                          # (M,3)
        X_h   = np.hstack([X_vis, np.ones((len(X_vis),1))])  # (M,4)
        x_h   = (P @ X_h.T).T                         # (M,3)
        x_proj = x_h[:,:2] / x_h[:,2:3]               # (M,2)

        # get observed pixels
        x_obs = np.array([tracks[j][i] for j in visible_js])  # (M,2)

        # residuals
        res.append((x_obs - x_proj).ravel())           # (2M,)

    return np.concatenate(res) if res else np.array([])


def BundleAdjustment(
    K:      np.ndarray,
    Cset:   list,
    Rset:   list,
    X:      np.ndarray,
    tracks: list,
    V:      np.ndarray,
    max_nfev: int = 100
):
    """
    Refine all camera poses and 3D points jointly by minimizing reprojection error.

    Inputs:
      K:        (3,3) intrinsics (fixed — not optimized)
      Cset:     list of I camera centers, each (3,)
      Rset:     list of I rotation matrices, each (3,3)
      X:        (J,3) 3D points
      tracks:   list of J dicts, tracks[j][i] = (u,v) observed in camera i
      V:        (I,J) visibility matrix from BuildVisibilityMatrix
      max_nfev: max optimizer function evaluations

    Returns:
      Cset_opt: list of I refined camera centers
      Rset_opt: list of I refined rotation matrices
      X_opt:    (J,3) refined 3D points
    """
    I = len(Cset)
    J = X.shape[0]
    assert V.shape == (I, J)

    p0 = pack(Cset, Rset, X)

    result = least_squares(
        residuals,
        p0,
        args=(K, V, tracks, I, J),
        method='trf',
        max_nfev=max_nfev
    )

    Cset_opt, qset_opt, X_opt = unpack(result.x, I, J)
    Rset_opt = [quat_to_rotmat(q) for q in qset_opt]

    return Cset_opt, Rset_opt, X_opt