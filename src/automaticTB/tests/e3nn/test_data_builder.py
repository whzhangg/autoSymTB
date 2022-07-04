from automaticTB.atomic_orbitals import Orbitals, OrbitalsList
from automaticTB.structure import Site
from automaticTB.SALCs import LinearCombination, VectorSpace
import numpy as np
import typing
from automaticTB.utilities import Pair
from automaticTB.e3nn.MOadoptor import get_geometric_data_from_pair_linear_combination

def get_line_molecular() -> typing.List[LinearCombination]:
    sites = [
        Site(1, np.array([-1,0,0])),
        Site(1, np.array([ 0,0,0])),
        Site(1, np.array([ 1,0,0]))
    ]
    orbitals = OrbitalsList(
        [Orbitals([0]), Orbitals([0]), Orbitals([0])]
    )
    vectorspace = VectorSpace.from_sites_and_orbitals(sites, orbitals)
    return vectorspace.get_nonzero_linear_combinations()

def test_data():
    lcs = get_line_molecular()
    apair = Pair(lcs[0], lcs[1])
    geometric = get_geometric_data_from_pair_linear_combination(apair)
    print(geometric)

test_data()

