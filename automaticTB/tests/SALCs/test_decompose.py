from automaticTB.linear_combination import Site, LinearCombination
from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.SALCs.decompose import decompose_vectorspace
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup
import numpy as np

symmetry = [np.eye(3), np.array([[-1,0,0], [0,1,0], [0,0,1]])]
group = SiteSymmetryGroup.from_cartesian_matrices(symmetry)

sites = [Site(1,np.array([1.0,0.0,0.0])), Site(1,np.array([-1.0,0.0,0.0]))]
vectorspace = VectorSpace.from_sites(sites, starting_basis="s")

def test_decompose_m():
    subspaces = decompose_vectorspace(vectorspace, group)
    for key, vspace in subspaces.items():
        print(key)
        vspace.remove_linear_dependent()
        for lc in vspace.get_linear_combinations():
            print(lc.pretty_str)

