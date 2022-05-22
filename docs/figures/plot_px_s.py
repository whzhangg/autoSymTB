from automaticTB.SALCs import LinearCombination
from automaticTB.structure import Site
from automaticTB.atomic_orbitals import Orbitals, OrbitalsList
import numpy as np
from automaticTB.utilities import random_rotation_matrix
from automaticTB.plotting.plot_LC import make_plot_normalized_LC

def get_px(y, px1, px2):
    sites = [
        Site(1, np.array([-1, 0, y])),
        Site(1, np.array([ 0, 0, y])),
        Site(1, np.array([ 1, 0, y])),
    ]
    orbitals = [Orbitals([1]), Orbitals([0]), Orbitals([1])]
    coefficients = [0,0,px1,1,0,0,px2]
    return sites, orbitals, coefficients

def get_test_linearcombination():
    sites = []
    orbitals = []
    coefficients = []
    s,o,c = get_px(1.5, 0, 1)
    sites += s
    orbitals += o
    coefficients += c
    s,o,c = get_px(0.5, 1, 0)
    sites += s
    orbitals += o
    coefficients += c
    
    s,o,c = get_px(-0.5, 1, 1)
    sites += s
    orbitals += o
    coefficients += c
    
    s,o,c = get_px(-1.5,-1, 1)
    sites += s
    orbitals += o
    coefficients += c
    return LinearCombination(sites, OrbitalsList(orbitals), np.array(coefficients, dtype = float))

def test_that_rotation_is_correct_by_printing():
    lc = get_test_linearcombination()  
    make_plot_normalized_LC(lc, "px-s.pdf")

test_that_rotation_is_correct_by_printing()