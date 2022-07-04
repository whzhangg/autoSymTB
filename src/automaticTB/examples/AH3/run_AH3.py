from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.interaction import get_free_interaction_AO, InteractionEquation
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.AH3.structure import get_nncluster_AH3
from automaticTB.printing import print_ao_pairs
from automaticTB.plotting import DensityCubePlot

def find_MOs_AH3():
    print("Solving for AH3 molecular ...")
    nncluster = get_nncluster_AH3()
    vectorspace = VectorSpace.from_NNCluster(nncluster)

    named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
    print("Symmetry adopted Linear combinations ...")
    for nlc in named_lcs:
        print(nlc.name)
        print(nlc.lc)

    free_pairs = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs).free_AOpairs
    print_ao_pairs(nncluster, free_pairs)

def plot_MOs_AH3():
    nncluster = get_nncluster_AH3()
    vectorspace = VectorSpace.from_NNCluster(nncluster)

    named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
    print("Symmetry adopted Linear combinations ...")
    for nlc in named_lcs:
        plot = DensityCubePlot.from_linearcombinations([nlc.lc], quality="high")
        plot.plot_to_file(f"{str(nlc.name)}.cube")


if __name__ == "__main__":
    plot_MOs_AH3()