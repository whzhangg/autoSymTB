from automaticTB.rotation import print_matrix, orbital_rotation_from_symmetry_matrix, rotate_linear_combination_from_symmetry_matrix
from automaticTB.linear_combination import Site, LinearCombination
import numpy as np
from automaticTB.utilities import random_rotation_matrix
from automaticTB.plotting.plot_LC import make_plot_normalized_LC

def print_rotation_matrix():
    angle = np.pi / 4
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

    combined = rotation_x @ rotation_y

    print(np.linalg.inv(combined))
    print("along x")
    print_matrix(orbital_rotation_from_symmetry_matrix(rotation_x), "{:>6.3f}")
    print("along y")
    print_matrix(orbital_rotation_from_symmetry_matrix(rotation_y), "{:>6.3f}")
    print("along z")
    print_matrix(orbital_rotation_from_symmetry_matrix(rotation_z), "{:>6.3f}")


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

coefficients = np.eye(9)


def show_that_rotation_is_correct():
    lc = LinearCombination(sites, coefficients)    
    make_plot_normalized_LC(lc, "unrotated.pdf")
    rotated = rotate_linear_combination_from_symmetry_matrix(lc, random_rotation_matrix())
    make_plot_normalized_LC(rotated, "rotated.pdf")

show_that_rotation_is_correct()