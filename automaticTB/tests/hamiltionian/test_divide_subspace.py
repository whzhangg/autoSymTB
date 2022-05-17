from automaticTB.structure.sites import Site
from automaticTB.structure.nncluster import CrystalSites_and_Equivalence
from automaticTB.SALCs.linear_combination import Orbitals
from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.SALCs.decompose import decompose_vectorspace
from automaticTB.hamiltionian.MOcoefficient import MOCoefficient
from automaticTB.hamiltionian.interaction import RandomInteraction
from automaticTB.hamiltionian.NNInteraction import get_interaction_matrix_from_MO_and_interaction
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup, get_point_group_as_SiteSymmetryGroup
import numpy as np
import typing

def test_dividesubspace_C3v():
    w = np.sqrt(3) / 2
    group = get_point_group_as_SiteSymmetryGroup("3m")  # D3h
    sites = [ Site(1, np.array([0.0, 0.0, 0.0])),
              Site(1,np.array([0.0,1.0,0.0])), 
              Site(1,np.array([w, -0.5,0.0])), 
              Site(1,np.array([-1.0 * w, -0.5, 0.0]))]
    basis = [Orbitals("s"),Orbitals("s"),Orbitals("s"),Orbitals("s")]
    vectorspace = VectorSpace.from_sites(sites, basis)
    equivalent = {
        0: {0}, 1: {1,2,3}
    }
    subspaces = decompose_vectorspace(vectorspace, group)
    site_sym = CrystalSites_and_Equivalence(sites, equivalent)

    mo = MOCoefficient(site_sym, subspaces)
    #print(mo.matrixA)
    aolist, hij =get_interaction_matrix_from_MO_and_interaction(mo, RandomInteraction())
    print(aolist)
    print(hij)

if __name__ == "__main__":
    test_dividesubspace_C3v()