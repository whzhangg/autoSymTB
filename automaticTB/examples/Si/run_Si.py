from automaticTB.SALCs import VectorSpace
from automaticTB.SALCs.stable_decompose import decompose_vectorspace_to_namedLC
from automaticTB.interaction import get_free_interaction_AO, InteractionEquation
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.Si.structure import get_si_structure
from automaticTB.printing import print_ao_pairs
from automaticTB.tools import Pair

def find_free_interaction_for_Si_nncluster():
    print("Solving for the free nearest neighbor interaction in Si")
    print("Starting ...")
    structure = get_si_structure()
    nncluster = structure.nnclusters[0]

    vectorspace = VectorSpace.from_NNCluster(nncluster)
    print("Solve vectorspace ... ")
    named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
    print("Solve Interaction ...")
    free_pairs = get_free_interaction_AO(nncluster, named_lcs, debug=False)
    ao = nncluster.AOlist
    free_pairs = [
        Pair(ao[fp.left], ao[fp.right]) for fp in free_pairs
    ]
    print_ao_pairs(nncluster, free_pairs)

def find_free_interaction_for_Si_nncluster_new():
    print("Solving for the free nearest neighbor interaction in Si")
    print("Starting ...")
    structure = get_si_structure()
    nncluster = structure.nnclusters[0]

    vectorspace = VectorSpace.from_NNCluster(nncluster)
    print("Solve vectorspace ... ")
    named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
    print("Solve Interaction ...")

    system = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
    free_pairs = system.free_AOpairs
    
    print_ao_pairs(nncluster, free_pairs)

if __name__ == "__main__":
    find_free_interaction_for_Si_nncluster_new()