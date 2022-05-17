from automaticTB.structure.sites import Site
from automaticTB.SALCs.linear_combination import Orbitals
from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.SALCs.decompose import decompose_vectorspace
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup, get_point_group_as_SiteSymmetryGroup
import numpy as np
import typing


def get_vectorspace_group_m() -> typing.Tuple[VectorSpace, SiteSymmetryGroup]:
    symmetry = [np.eye(3), np.array([[-1,0,0], [0,1,0], [0,0,1]])]
    group = SiteSymmetryGroup.from_cartesian_matrices(symmetry)

    sites = [Site(1,np.array([1.0,0.0,0.0])), Site(1,np.array([-1.0,0.0,0.0]))]
    basis = [Orbitals("s"),Orbitals("s")]
    vectorspace = VectorSpace.from_sites(sites, basis)
    return vectorspace, group

def get_vectorspace_group_C3v() -> typing.Tuple[VectorSpace, SiteSymmetryGroup]:
    w = np.sqrt(3) / 2
    group = get_point_group_as_SiteSymmetryGroup("3m")  # D3h
    sites = [ Site(1, np.array([0.0, 0.0, 0.0])),
              Site(1,np.array([0.0,1.0,0.0])), 
              Site(1,np.array([w, -0.5,0.0])), 
              Site(1,np.array([-1.0 * w, -0.5, 0.0]))]
    basis = [Orbitals("s"),Orbitals("s"),Orbitals("s"),Orbitals("s")]
    vectorspace = VectorSpace.from_sites(sites, basis)
    return vectorspace, group


def test_decompose_m():
    vectorspace, group = get_vectorspace_group_m()
    subspaces = decompose_vectorspace(vectorspace, group)
    for key, vspace in subspaces.items():
        print(key)
        for lc in vspace.get_nonzero_linear_combinations():
            print(lc)

def test_decompose_C3v():
    vectorspace, group = get_vectorspace_group_C3v()
    subspaces = decompose_vectorspace(vectorspace, group)
    for key, vspace in subspaces.items():
        print(key)
        for lc in vspace.get_nonzero_linear_combinations():
            print(lc)

if __name__ == "__main__":
    test_decompose_m()
    test_decompose_C3v()