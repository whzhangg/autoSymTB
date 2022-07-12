from automaticTB.rotation import orbital_rotation_from_symmetry_matrix
from automaticTB.utilities import random_rotation_matrix
from automaticTB.SALCs import LinearCombination
from automaticTB.structure import Site
from automaticTB.atomic_orbitals import Orbitals, OrbitalsList
from automaticTB.parameters import LC_coefficients_dtype
import numpy as np


def test_lc_rotation():
    sites = [
        Site(1, np.array([1.0, 0.0, 0.0])),
        Site(1, np.array([-1.0, 0.0, 0.0]))
    ]
    ol = OrbitalsList(
        [Orbitals([0]), Orbitals([0])]
    )
    coefficient = np.array([1.0, 0.0], dtype =LC_coefficients_dtype)
    lc = LinearCombination(sites, ol, coefficient)

    randomly_rotated = lc.general_rotate(random_rotation_matrix())
    assert np.allclose(randomly_rotated.coefficients, coefficient)

    reflection = np.array( [[-1.0, 0, 0], [0,1,0], [0,0,1]])
    symmetry_rotated = lc.symmetry_rotate(reflection)
    assert np.allclose(symmetry_rotated.coefficients, np.array([0.0, 1.0], dtype =LC_coefficients_dtype))
