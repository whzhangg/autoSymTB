from automaticTB.SALCs import LinearCombination
from automaticTB.structure import Site
from automaticTB.atomic_orbitals import Orbitals, OrbitalsList
from automaticTB.plotting import DensityCubePlot
import numpy as np
from automaticTB.tools import random_rotation_matrix

from automaticTB.rotation import orbital_rotation_from_symmetry_matrix


def test_rotation_matrix_direction_corret():
    angle = np.random.rand()
    cos = np.cos(angle)
    sin = np.sin(angle)

    rotation_x = np.array([[1,   0,    0],
                           [0, cos, -sin],
                           [0, sin,  cos]])
    rotation_y = np.array([[ cos, 0, sin],
                           [   0, 1,   0],
                           [-sin, 0, cos]])
    rotation_z = np.array([[cos, -sin, 0], 
                           [sin,  cos, 0],
                           [  0,    0, 1]])

    # test p orbital orientation is correct (Py, Pz, Px)
    # along x
    rot_mat = orbital_rotation_from_symmetry_matrix(rotation_x, "1x0e + 1x1o + 1x2e")
    assert np.isclose(rot_mat[3,3], 1.0), "px"

    # along y
    rot_mat = orbital_rotation_from_symmetry_matrix(rotation_y, "1x0e + 1x1o + 1x2e")
    assert np.isclose(rot_mat[1,1], 1.0), "py"

    # along z
    rot_mat = orbital_rotation_from_symmetry_matrix(rotation_z, "1x0e + 1x1o + 1x2e")
    assert np.isclose(rot_mat[2,2], 1.0), "pz"
    assert np.isclose(rot_mat[6,6], 1.0), "dz2"


def get_test_linearcombination():
    sites = [
        Site(1, np.array([ 0, 0, 0])),
        Site(1, np.array([ 1, 1, 1])),
        Site(1, np.array([ 1,-1, 1])),
        Site(1, np.array([-1, 1, 1])),
        Site(1, np.array([-1,-1, 1])),
        Site(1, np.array([ 1, 1,-1])),
        Site(1, np.array([ 1,-1,-1])),
        Site(1, np.array([-1, 1,-1])),
        Site(1, np.array([-1,-1,-1]))
    ]
    orbitals = [Orbitals([0,1,2])] * 9
    coefficients = np.eye(9).flatten()
    return LinearCombination(sites, OrbitalsList(orbitals), coefficients)


def test_rotation_is_correct():
    lc = get_test_linearcombination()  
    cub_plot_unrotated = DensityCubePlot.from_linearcombinations([lc])
    cub_plot_unrotated.plot_to_file("unrotated.cube")
    rotated = lc.general_rotate(random_rotation_matrix())
    cub_plot_rotated = DensityCubePlot.from_linearcombinations([rotated])
    cub_plot_rotated.plot_to_file("rotated.cube")