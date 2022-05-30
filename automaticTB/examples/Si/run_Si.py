from automaticTB.SALCs import VectorSpace
from automaticTB.SALCs.stable_decompose import decompose_vectorspace_to_namedLC
from automaticTB.MOInteraction import get_free_interaction_AO
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.Si.structure import get_si_structure
from automaticTB.examples.utilities import print_ao_pairs

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
    
    print_ao_pairs(nncluster.AOlist, free_pairs)

if __name__ == "__main__":
    find_free_interaction_for_Si_nncluster()