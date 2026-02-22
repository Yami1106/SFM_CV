# Phase1/ExtractCameraPose.py
import numpy as np

def ExtractCameraPose(E: np.ndarray):
    """
      The four configurations:
        C1 =  U[:,3]   and R1 = U W  V^T
        C2 = -U[:,3]   and R2 = U W  V^T
        C3 =  U[:,3]   and R3 = U W^T V^T
        C4 = -U[:,3]   and R4 = U W^T V^T

    Inputs:
      E: (3,3) essential matrix

    Returns:
      Cset: list of 4 camera centers, each shape (3,)
      Rset: list of 4 rotations, each shape (3,3)
    """
    assert E.shape == (3, 3)

    # 1) SVD of E
    U, _, Vt = np.linalg.svd(E)

    # 2) make sure U and V correspond to a proper rotation no reflection
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # 3) Define W 
    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float64)

    # 4) Candidate rotations
    # R = U W V^T  and  R = U W^T V^T
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # 5) Candidate camera centers 
    C = U[:, 2]

    # 6) Build the four combinations
    Cset = [ C, -C,  C, -C ]
    Rset = [R1,  R1, R2,  R2]

    # 7) Ensure det(R) = +1 for each rotation.
    # If det(R) is -1, multiply both R and C by -1 for that candidate.
    for i in range(4):
        if np.linalg.det(Rset[i]) < 0:
            Rset[i] = -Rset[i]
            Cset[i] = -Cset[i]

    # 8) Return as lists
    Cset = [c.reshape(3,) for c in Cset]
    Rset = [R.reshape(3,3) for R in Rset]

    return Cset, Rset
