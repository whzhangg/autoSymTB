from automaticTB.linear_combination import Site
from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.SALCs.decompose import decompose_vectorspace
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup, get_point_group_as_SiteSymmetryGroup
import numpy as np
import typing
from automaticTB.tightbinding.lc_product import RepLabelledBasis

def get_vectorspace_group_C3v() -> typing.Tuple[VectorSpace, SiteSymmetryGroup]:
    w = np.sqrt(3) / 2
    group = get_point_group_as_SiteSymmetryGroup("3m")  # D3h
    sites = [ Site(1, np.array([0.0, 0.0, 0.0])),
              Site(1,np.array([0.0,1.0,0.0])), 
              Site(1,np.array([w, -0.5,0.0])), 
              Site(1,np.array([-1.0 * w, -0.5, 0.0]))]
    vectorspace = VectorSpace.from_sites(sites, starting_basis="s")
    return vectorspace, group

class RandomInteraction:
    def __init__(self) -> None:
        self.interaction_dict = {}
        self.random1 = np.random.rand()
        self.random2 = np.random.rand()

    def get_interaction(self, s1: str, s2: str):
        if s1 != s2: 
            return 0.0
        if "E->" in s1:
            return self.random1
        else:
            return self.random2

def all_value_equal(inputlist: typing.List[float]) -> bool:
    return np.all(np.isclose(np.array(inputlist), inputlist[0]))

def test_formulate_equation_system_C3v():
    vectorspace, group = get_vectorspace_group_C3v()
    subspaces = decompose_vectorspace(vectorspace, group)
    lcs = RepLabelledBasis.from_decomposed_vectorspace(subspaces)
    b = []

    random_interaction = RandomInteraction()
    for i in range(lcs.rank):
        pairedLC = lcs.get_pair_LCs(i)
        if pairedLC.rep1 != pairedLC.rep2:
            b.append(0.0)
        else:
            b.append(random_interaction.get_interaction(pairedLC.rep1, pairedLC.rep2))
    #print_matrix(lcs.matrixA, "{:>6.3}")
    invA = np.linalg.inv(lcs.matrixA)
    x = invA.dot(np.array(b))
    x = x.reshape((4,4))

    # show that matrix is of the form:
    # a, b, b, b
    # b, c, d, d
    # b, d, c, d
    # b, d, d, c
    assert all_value_equal([x[1,1], x[2,2], x[3,3]])
    assert all_value_equal([x[0,1], x[0,2], x[0,3],
                            x[1,0], x[2,0], x[3,0]])
    assert all_value_equal([x[1,2], x[1,3], x[2,3],
                            x[2,1], x[3,1], x[3,2]])
    