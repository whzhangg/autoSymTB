import numpy as np
import torch
from .linear_combination import LinearCombination
from e3nn import o3


def rotate_linear_combination_from_matrix(input_lc: LinearCombination, cartesian_rotation: np.ndarray) \
-> LinearCombination:
    # arbitrary rotation, new sites does not necessary correspond to the old ones
    new_sites, rotated_coefficients = \
        _get_new_sites_and_rotated_coefficient(input_lc, cartesian_rotation)
    return LinearCombination(new_sites, input_lc.orbitals, rotated_coefficients)


def rotate_linear_combination_from_symmetry_matrix(input_lc: LinearCombination, cartesian_rotation: np.ndarray) \
-> LinearCombination:
    # symmetry rotation, atomic site sequence are reorganized so that they are the same as input
    new_sites, rotated_coefficients = \
        _get_new_sites_and_rotated_coefficient(input_lc, cartesian_rotation)

    permute_index = [
        new_sites.index(site) for site in input_lc.sites
    ]
    return LinearCombination(input_lc.sites, input_lc.orbitals, rotated_coefficients[permute_index])



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


def _orbital_rotation_from_symmetry_matrix(cartesian_symmetry_operation: np.ndarray, aoirrep: str) \
-> np.ndarray:
    irrep = o3.Irreps(aoirrep)
    changed = _rot_to_change_x_y(cartesian_symmetry_operation)
    rotation_torch: torch.Tensor = irrep.D_from_matrix(torch.from_numpy(changed))
    return rotation_torch.numpy()


def _get_new_sites_and_rotated_coefficient(input_lc: LinearCombination, cartesian_rotation: np.ndarray):
    new_sites = [
        site.rotate(cartesian_rotation) for site in input_lc.sites
    ]
    orbital_rotation = _orbital_rotation_from_symmetry_matrix(cartesian_rotation, input_lc.orbitals.irreps_str)
    rotated_coefficients = np.einsum("ij, kj -> ki", orbital_rotation, input_lc.coefficients)
    return new_sites, rotated_coefficients

