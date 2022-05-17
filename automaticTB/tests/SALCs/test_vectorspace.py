from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.structure.sites import Site
from automaticTB.SALCs.linear_combination import Orbitals
import numpy as np

sites = [
    Site(1, np.array([ 1,0,0])),
    Site(1, np.array([-1,0,0])),
    Site(1, np.array([0,1,0])),
    Site(1, np.array([0,-1,0])),
    Site(1, np.array([0,0,1])),
    Site(1, np.array([0,0,-1])),
]
orbitals = [
    Orbitals("s"),
    Orbitals("p"),
    Orbitals("d"),
    Orbitals("s p"),
    Orbitals("s d"),
    Orbitals("p d")
]

def test_initiating_vectorspace():
    vectorspace = VectorSpace.from_sites(sites, orbitals)
    assert vectorspace.rank == 27
    assert vectorspace.nbasis == 27

def test_initiating_vectorspace_from_linearcombination():
    vectorspace = VectorSpace.from_sites(sites, orbitals)
    lcs = vectorspace.get_nonzero_linear_combinations()
    vectorspace_rebuilt = VectorSpace.from_list_of_linear_combinations(lcs)

    assert np.allclose(
        vectorspace._vector_coefficients, vectorspace_rebuilt._vector_coefficients
    )



"""
def test_initiating_from_sites_s():
    vectorspace = VectorSpace.from_sites(sites, starting_basis="s")
    assert vectorspace.rank == len(sites) * 1

def test_initiating_from_sites_p():
    vectorspace = VectorSpace.from_sites(sites, starting_basis="p")
    assert vectorspace.rank == len(sites) * 3

def test_initiating_from_sites_sp():
    vectorspace = VectorSpace.from_sites(sites, starting_basis="s p")
    assert vectorspace.rank == len(sites) * 4

def test_initiating_from_sites_d():
    vectorspace = VectorSpace.from_sites(sites, starting_basis="d")
    assert vectorspace.rank == len(sites) * 5

def test_initiating_from_linear_combinations():
    vectorspace = VectorSpace.from_sites(sites, starting_basis="s p d")
    lcs = vectorspace.get_nonzero_linear_combinations()
    vectorspace_built = VectorSpace.from_list_of_linear_combinations(lcs)
    
    assert vectorspace._sites == vectorspace_built._sites
    assert np.allclose(
        vectorspace._vector_coefficients, vectorspace_built._vector_coefficients)

"""