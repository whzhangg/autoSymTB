from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.linear_combination import Site
import numpy as np

sites = [
    Site(1, np.array([ 1,0,0])),
    Site(1, np.array([-1,0,0])),
    Site(1, np.array([0,1,0])),
    Site(1, np.array([0,-1,0])),
    Site(1, np.array([0,0,1])),
    Site(1, np.array([0,0,-1])),
]

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
    lcs = vectorspace.get_linear_combinations()
    vectorspace_built = VectorSpace.from_list_of_linear_combinations(lcs)
    
    assert vectorspace._sites == vectorspace_built._sites
    assert np.allclose(
        vectorspace._vector_coefficients, vectorspace_built._vector_coefficients)