from automaticTB.examples.structures import fcc_si
from automaticTB.structure import Structure
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.printing import print_namedLCs, print_ao_pairs
from automaticTB.interaction import get_free_interaction_AO, InteractionEquation
from automaticTB.tools import Pair

def test_decompose():
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    struct = Structure.from_cpt_rcut(c,p,t, {"Si": [0,1]}, 3.0)
    nncluster = struct.nnclusters[0]
    vectorspace = VectorSpace.from_NNCluster(nncluster)
    print("Solve vectorspace ... ")
    named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
    print_namedLCs(named_lcs)
    print("\n".join([str(nlc.name) for nlc in named_lcs]))

def test_decompose_sps():
    """
    to this step, it is OK to add 
    """
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    struct = Structure.from_cpt_rcut(c,p,t, {"Si": [0,1,0]}, 3.0)
    nncluster = struct.nnclusters[0]
    vectorspace = VectorSpace.from_NNCluster(nncluster)
    print("Solve vectorspace ... ")
    named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
    print_namedLCs(named_lcs)
    print("\n".join([str(nlc.name) for nlc in named_lcs]))


def test_decompose_sps():
    """
    to this step, it is OK to add 
    """
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    struct = Structure.from_cpt_rcut(c,p,t, {"Si": [0,1,0]}, 3.0)
    nncluster = struct.nnclusters[0]
    vectorspace = VectorSpace.from_NNCluster(nncluster)
    print("Solve vectorspace ... ")
    named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
    print_namedLCs(named_lcs)
    print("\n".join([str(nlc.name) for nlc in named_lcs]))


def test_find_free_interaction():
    """
    to this step, it is OK to add 
    """
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    struct = Structure.from_cpt_rcut(c,p,t, {"Si": [0,1,0]}, 3.0)
    nncluster = struct.nnclusters[0]
    vectorspace = VectorSpace.from_NNCluster(nncluster)
    print("Solve vectorspace ... ")
    named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
    print("\n".join([str(nlc.name) for nlc in named_lcs]))
    equation = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
    free_pairs = equation.free_AOpairs
    print_ao_pairs(nncluster, free_pairs)
    print("Solve Interaction ...")
    free_pairs = get_free_interaction_AO(nncluster, named_lcs, debug=False)
    ao = nncluster.AOlist
    free_pairs = [
        Pair(ao[fp.left], ao[fp.right]) for fp in free_pairs
    ]
    print_ao_pairs(nncluster, free_pairs)

def test_subspace_pairs():
    """
    to this step, it is OK to add 
    """
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    struct = Structure.from_cpt_rcut(c,p,t, {"Si": [0,1]}, 3.0)
    nncluster = struct.nnclusters[0]
    vectorspace = VectorSpace.from_NNCluster(nncluster)
    for pair in nncluster.interaction_subspace_pairs:
        print(pair)

test_subspace_pairs()