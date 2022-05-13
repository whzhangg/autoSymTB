from automaticTB.rotation import rotate_linear_combination_from_matrix
from automaticTB.linear_combination import Site, Orbitals, LinearCombination
import numpy as np
from automaticTB.utilities import random_rotation_matrix
from automaticTB.plotting.plot_LC import make_plot_normalized_LC


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


def test_that_rotation_is_correct_by_printing():
    lc = LinearCombination(sites, Orbitals("s p d"), np.eye(9))    
    make_plot_normalized_LC(lc, "unrotated.pdf")
    rotated = rotate_linear_combination_from_matrix(lc, random_rotation_matrix())
    make_plot_normalized_LC(rotated, "rotated.pdf")

if __name__ == "__main__":
    test_that_rotation_is_correct_by_printing()