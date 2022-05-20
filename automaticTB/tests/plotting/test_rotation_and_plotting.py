from automaticTB.SALCs import LinearCombination
from automaticTB.structure import Site
from automaticTB.atomic_orbitals import Orbitals, OrbitalsList
import numpy as np
from automaticTB.utilities import random_rotation_matrix
from automaticTB.plotting.plot_LC import make_plot_normalized_LC


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

def test_that_rotation_is_correct_by_printing():
    lc = get_test_linearcombination()  
    make_plot_normalized_LC(lc, "unrotated.pdf")
    rotated = lc.general_rotate(random_rotation_matrix())
    make_plot_normalized_LC(rotated, "rotated.pdf")
