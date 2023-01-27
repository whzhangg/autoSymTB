import numpy as np
import torch
from e3nn import o3

rotat_x = np.array([[0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])

inv_rot = np.array([[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0]])


def orbital_rotation_from_symmetry_matrix(
    cartesian_symmetry_operation: np.ndarray, aoirrep: str
) -> np.ndarray:
    """convert a cartesian symmetry operation into corresponding orbital rotation"""
    irrep = o3.Irreps(aoirrep)
    changed = inv_rot @ cartesian_symmetry_operation @ rotat_x
    rotation_torch = irrep.D_from_matrix(torch.from_numpy(changed))
    return rotation_torch.numpy()
