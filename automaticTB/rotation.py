import numpy as np
import torch
from automaticTB.linear_combination import AOLISTS, aoirrep, LinearCombination
from e3nn import o3

def print_matrix(m: np.ndarray, format:str):
    assert len(m.shape) == 2
    result = ""
    for row in m:
        for cell in row:
            result += " " + format.format(cell)
        result += "\n"
    print(result)

def rot_to_change_x_y(input: np.ndarray) -> np.ndarray:
    # 1,-1 -> py
    # 1, 0 -> pz
    # 1, 1 -> px, 
    # rotation_x permute the axis so that z->x, x->y, y->z
    # this is a bit confusion, but works 
    rotation_x = np.array([[0, 0, 1],
                           [1, 0, 0],
                           [0, 1, 0]])
    inv_rot = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
    return inv_rot @ input @ rotation_x

def orbital_rotation_from_symmetry_matrix(cartesian_symmetry_operation: np.ndarray) \
-> np.ndarray:
    irrep = o3.Irreps(aoirrep)
    changed = rot_to_change_x_y(cartesian_symmetry_operation)
    rotation_torch: torch.Tensor = irrep.D_from_matrix(torch.from_numpy(changed))
    return rotation_torch.numpy()

def rotate_linear_combination_from_symmetry_matrix(
    input_lc: LinearCombination, cartesian_rotation: np.ndarray) \
-> LinearCombination:
    new_sites = [
        site.rotate(cartesian_rotation) for site in input_lc.sites
    ]
    orbital_rotation = orbital_rotation_from_symmetry_matrix(cartesian_rotation)
    rotated_coefficients = np.einsum("ij, kj -> ki", orbital_rotation, input_lc.coefficients)
    return LinearCombination(new_sites, rotated_coefficients)
    