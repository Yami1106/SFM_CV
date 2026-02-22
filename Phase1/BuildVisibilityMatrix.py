import numpy as np


def BuildVisibilityMatrix(tracks, num_cameras: int, num_points: int) -> np.ndarray:
    """
    Build visibility matrix V of shape (num_cameras, num_points).

    V[i, j] = 1  if 3D point j is observed by camera i
    V[i, j] = 0  anything else

    Inputs:
      tracks:      list of length num_points
                   tracks[j] is a dict mapping camera_id (0-based) → (u, v)
      num_cameras: number of cameras I
      num_points:  number of 3D points J

    Returns:
      V: (I, J) binary array
    """
    assert len(tracks) == num_points

    V = np.zeros((num_cameras, num_points), dtype=np.int32)

    for j, obs in enumerate(tracks):
        if obs is None:
            continue
        for cam_id in obs.keys():
            if 0 <= cam_id < num_cameras:
                V[cam_id, j] = 1

    return V