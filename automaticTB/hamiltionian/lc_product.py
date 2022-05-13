from automaticTB.linear_combination import Site, LinearCombination
from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.SALCs.decompose import decompose_vectorspace_onelevel, decompose_vectorspace_recursive, decompose_vectorspace
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup, get_point_group_as_SiteSymmetryGroup
import numpy as np
import typing

def get_vectorspace_group_C3v() -> typing.Tuple[VectorSpace, SiteSymmetryGroup]:
    w = np.sqrt(3) / 2
    group = get_point_group_as_SiteSymmetryGroup("3m")  # D3h
    sites = [ Site(1, np.array([0.0, 0.0, 0.0])),
              Site(1,np.array([0.0,1.0,0.0])), 
              Site(1,np.array([w, -0.5,0.0])), 
              Site(1,np.array([-1.0 * w, -0.5, 0.0]))]
    vectorspace = VectorSpace.from_sites(sites, starting_basis="s")
    return vectorspace, group

def decompose_C3v() -> typing.List[LinearCombination]:
    vectorspace, group = get_vectorspace_group_C3v()
    subspaces = decompose_vectorspace(vectorspace, group)
    lcs = []
    for key, vspace in subspaces.items():
        lcs += vspace.get_nonzero_linear_combinations()
    return lcs

def formulate_equation_system():
    # this basically produce the desired linear equation
    lcs = decompose_C3v()
    A = []
    b = []
    for lc1 in lcs:
        for lc2 in lcs:
            tp = np.tensordot(lc1.coefficients.flatten(), lc2.coefficients.flatten(), axes = 0).flatten()
            A.append(tp)
            print(tp)
            b.append(1.0)
            # https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html

if __name__ == "__main__":
    formulate_equation_system()