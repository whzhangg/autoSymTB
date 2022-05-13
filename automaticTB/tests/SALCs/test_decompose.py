from automaticTB.linear_combination import Site, LinearCombination
from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.SALCs.decompose import decompose_vectorspace_onelevel, decompose_vectorspace_recursive, decompose_vectorspace
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup, get_point_group_as_SiteSymmetryGroup
import numpy as np
import typing


def get_vectorspace_group_m() -> typing.Tuple[VectorSpace, SiteSymmetryGroup]:
    symmetry = [np.eye(3), np.array([[-1,0,0], [0,1,0], [0,0,1]])]
    group = SiteSymmetryGroup.from_cartesian_matrices(symmetry)

    sites = [Site(1,np.array([1.0,0.0,0.0])), Site(1,np.array([-1.0,0.0,0.0]))]
    vectorspace = VectorSpace.from_sites(sites, starting_basis="s")
    return vectorspace, group

def get_vectorspace_group_C3v() -> typing.Tuple[VectorSpace, SiteSymmetryGroup]:
    w = np.sqrt(3) / 2
    group = get_point_group_as_SiteSymmetryGroup("3m")  # D3h
    sites = [ Site(1, np.array([0.0, 0.0, 0.0])),
              Site(1,np.array([0.0,1.0,0.0])), 
              Site(1,np.array([w, -0.5,0.0])), 
              Site(1,np.array([-1.0 * w, -0.5, 0.0]))]
    vectorspace = VectorSpace.from_sites(sites, starting_basis="s")
    return vectorspace, group


def test_decompose_m_onelevel():
    vectorspace, group = get_vectorspace_group_m()
    subspaces = decompose_vectorspace_onelevel(vectorspace, group)
    for key, vspace in subspaces.items():
        print(key)
        vspace.remove_linear_dependent()
        for lc in vspace.get_nonzero_linear_combinations():
            print(lc.pretty_str)

def test_decompose_m():
    vectorspace, group = get_vectorspace_group_m()
    subspaces = decompose_vectorspace(vectorspace, group)
    for key, vspace in subspaces.items():
        print(key)
        vspace.remove_linear_dependent()
        for lc in vspace.get_nonzero_linear_combinations():
            print(lc.pretty_str)

def test_decompose_C3v_onelevel():
    vectorspace, group = get_vectorspace_group_C3v()
    subspaces = decompose_vectorspace_onelevel(vectorspace, group)
    for key, vspace in subspaces.items():
        print(key)
        vspace.remove_linear_dependent()
        for lc in vspace.get_nonzero_linear_combinations():
            if lc: print(lc.pretty_str)

def test_decompose_C3v():
    vectorspace, group = get_vectorspace_group_C3v()
    subspaces = decompose_vectorspace(vectorspace, group)
    for key, vspace in subspaces.items():
        print(key)
        for lc in vspace.get_nonzero_linear_combinations():
            if lc: 
                print(lc)
                print(lc.orbitals)
                print(lc.coefficients)

if __name__ == "__main__":
    test_decompose_C3v()