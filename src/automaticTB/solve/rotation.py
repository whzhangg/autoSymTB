import numpy as np
import torch
from e3nn import o3

rotation_x = np.array([[0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])

inv_rot = np.array([[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0]])


def _rot_to_change_x_y(input: np.ndarray) -> np.ndarray:
    # 1,-1 -> py
    # 1, 0 -> pz
    # 1, 1 -> px, 
    # rotation_x permute the axis so that z->x, x->y, y->z
    # this is a bit confusion, but works 
    rotation_x = np.array([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    inv_rot = np.array([[0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0]])
    return inv_rot @ input @ rotation_x


def orbital_rotation_from_symmetry_matrix(cartesian_symmetry_operation: np.ndarray, aoirrep: str) \
-> np.ndarray:
    irrep = o3.Irreps(aoirrep)
    changed = inv_rot @ cartesian_symmetry_operation @ rotation_x
    #changed = _rot_to_change_x_y(cartesian_symmetry_operation)
    rotation_torch: torch.Tensor = irrep.D_from_matrix(torch.from_numpy(changed))
    return rotation_torch.numpy()
