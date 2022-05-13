from automaticTB.linear_combination import LinearCombination, Site, Orbitals
from automaticTB.rotation import rotate_linear_combination_from_matrix, _orbital_rotation_from_symmetry_matrix
from automaticTB.utilities import random_rotation_matrix
import numpy as np


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
    rot_mat = _orbital_rotation_from_symmetry_matrix(rotation_x, "1x0e + 1x1o + 1x2e")
    assert np.isclose(rot_mat[3,3], 1.0), "px"

    # along y
    rot_mat = _orbital_rotation_from_symmetry_matrix(rotation_y, "1x0e + 1x1o + 1x2e")
    assert np.isclose(rot_mat[1,1], 1.0), "py"

    # along z
    rot_mat = _orbital_rotation_from_symmetry_matrix(rotation_z, "1x0e + 1x1o + 1x2e")
    assert np.isclose(rot_mat[2,2], 1.0), "pz"
    assert np.isclose(rot_mat[6,6], 1.0), "dz2"


sample_cluster_CH4 = [
    Site( 6, np.array([ 0, 0, 0]) ),
    Site( 1, np.array([ 1, 1, 1]) ),
    Site( 1, np.array([-1,-1, 1]) ),
    Site( 1, np.array([ 1,-1,-1]) ),
    Site( 1, np.array([-1, 1,-1]) )
]
coefficients = np.random.rand(5,9)

def test_rotation_can_run():

    rot = random_rotation_matrix()
    lc = LinearCombination(sample_cluster_CH4, Orbitals("s"), coefficients[:,0:1])
    rotate_linear_combination_from_matrix(lc, rot)

    lc = LinearCombination(sample_cluster_CH4, Orbitals("p"), coefficients[:,1:4])
    rotate_linear_combination_from_matrix(lc, rot)

    lc = LinearCombination(sample_cluster_CH4, Orbitals("d"), coefficients[:,4:9])
    rotate_linear_combination_from_matrix(lc, rot)

    lc = LinearCombination(sample_cluster_CH4, Orbitals("p d"), coefficients[:, 1:9])
    rotate_linear_combination_from_matrix(lc, rot)
