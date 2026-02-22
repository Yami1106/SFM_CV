# Phase1/EfromF.py
import numpy as np

def EfromF(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute Essential Matrix E from Fundamental Matrix F using intrinsic matrix K.

    Inputs:
      F: (3,3) Fundamental matrix
      K: (3,3) Intrinsic matrix

    Returns:
      E: (3,3) Essential matrix with corrected singular values
    """
    assert F.shape == (3, 3)
    assert K.shape == (3, 3)

    # 1) essential matrix from the formula
    E = K.T @ F @ K

    # 2) essential matrix constraints using SVD:
    U, S, Vt = np.linalg.svd(E)

    # 3) Fix orientation if needed:
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # 4) Force singular values to (1, 1, 0)
    S[-1] = 0

    # 5) Reconstruct corrected E
    E_corrected = U @ np.diag(S) @ Vt

    # 6) Optional: normalize E to a consistent scale (not required, but helps reproducibility)
    #    Example: normalize so ||E||_F = 1
    E_corrected = E_corrected / np.linalg.norm(E_corrected)

    return E_corrected
