from automaticTB.atomic_orbitals import Orbitals, OrbitalsList
from automaticTB.structure import Site
from automaticTB.SALCs import LinearCombination, VectorSpace
import numpy as np
import typing
from automaticTB.utilities import Pair
from automaticTB.e3nn.MOadoptor import get_geometric_data_from_pair_linear_combination
from automaticTB.e3nn.InvariantPolynomial import get_fitter, InvariantPolynomial

sites = [
        Site(1, np.array([-1,0,0])),
        Site(1, np.array([ 0,0,0])),
        Site(1, np.array([ 1,0,0]))
    ]
orbitals = OrbitalsList(
        [Orbitals([0]), Orbitals([0]), Orbitals([0])]
    )

def get_line_molecular() -> typing.List[LinearCombination]:
    vectorspace = VectorSpace.from_sites_and_orbitals(sites, orbitals)
    return vectorspace.get_nonzero_linear_combinations()

def get_pairs():
    lcs = get_line_molecular()
    pair1 = get_geometric_data_from_pair_linear_combination(Pair(lcs[0], lcs[1]))
    pair2 = get_geometric_data_from_pair_linear_combination(Pair(lcs[0], lcs[2]))

    return [pair1, pair2], [-1.0, 5.0]

def test_fit():
    lc1 = LinearCombination(
        sites, orbitals, np.array([1.0, 0.0, 1.0])
    ).get_normalized()

    lc2 = LinearCombination(
        sites, orbitals, np.array([1.0, 0.0,-1.0])
    ).get_normalized()

    lc3 = LinearCombination(
        sites, orbitals, np.array([0.0, 1.0, 0.0])
    )

    from automaticTB.e3nn.MOadoptor import embed_atomic_orbital_coefficients
    irr = embed_atomic_orbital_coefficients.irreps_str
    net = InvariantPolynomial(irr)

    result = net(get_geometric_data_from_pair_linear_combination(Pair(lc3, lc2)))
    print(result)

    result = net(get_geometric_data_from_pair_linear_combination(Pair(lc3, lc1)))
    print(result)

    result = net(get_geometric_data_from_pair_linear_combination(Pair(lc3, lc3)))
    print(result)

test_fit()