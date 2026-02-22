import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix, make_homogeneous

def get_error(F, x1, x2):
    x1_h = make_homogeneous(x1)
    x2_h = make_homogeneous(x2)
    Fx1  = F @ x1_h.T                        # (3,N)
    vals = np.sum(x2_h.T * Fx1, axis=0)      # (N,)  x2^T F x1
    return vals


def GetInliersRANSAC(
    x1: np.ndarray,
    x2: np.ndarray,
    M: int = 2000,
    epsilon: float = 0.05,
    seed: int = 0 # set this to get reproducible results
):
    """
    Get Inliers RANSAC

    Inputs:
      x1, x2 : (N,2) corresponding points
      M      : number of RANSAC iterations (loop i = 1..M)
      epsilon: threshold for inlier test |x1^T F x2| < epsilon

    Returns:
      inlier_mask: (N,) boolean array where True means inlier
    """
    assert x1.shape == x2.shape
    N = x1.shape[0]
    assert N >= 8

    rng = np.random.default_rng(seed)

    n_best = 0                
    best_inlier_mask = np.zeros(N, dtype=bool)  

    for i in range(M):
        # Choose 8 correspondences randomly
        idx = rng.choice(N, size=8, replace=False)
        x1_hat = x1[idx]
        x2_hat = x2[idx]

        # Estimate F from the sampled correspondences
        F = EstimateFundamentalMatrix(x1_hat, x2_hat)

        errs = get_error(F, x1, x2)
        inlier_mask = np.abs(errs) < epsilon
        n_inliers = int(np.sum(inlier_mask))

        if n_inliers > n_best:
            n_best = n_inliers
            best_inlier_mask = inlier_mask

    return best_inlier_mask
