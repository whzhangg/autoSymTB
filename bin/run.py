from automaticTB.tools import read_cif_to_cpt, valence_dictionary
from automaticTB.structure import Structure
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.interaction import InteractionEquation
from automaticTB.printing import print_ao_pairs, print_namedLCs
import os

def _get_structure(cif_filename: str, cutoff: float) -> Structure:
    if not os.path.exists(cif_filename):
        print(f"File {cif_filename} does not exist !")
        return

    c, p, t = read_cif_to_cpt(cif_filename)
    structure = Structure.from_cpt_rcut(c,p,t, valence_dictionary, cutoff)

    return structure

def get_free_parameters(cif_filename: str, cutoff: float) -> None:
    structure = _get_structure(cif_filename, cutoff)

    print( "> Solving for the free nearest neighbor interaction for structure ")
    print(f">           {cif_filename}")
    print( "> Starting ...")

    print(f"> There are {len(structure.types)} atoms in the primitive cell")
    for i, nncluster in enumerate(structure.nnclusters):
        print(f"> Site symmetry {nncluster.sitesymmetrygroup.groupname}")
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        print(f"> Solve vectorspace for atom {i+1} ")
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)

        system = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
        free_pairs = system.free_AOpairs
        print_ao_pairs(nncluster, free_pairs)
        print(" ")

def get_linearcombination(cif_filename: str, cutoff: float) -> None:
    structure = _get_structure(cif_filename, cutoff)
    print( "> Finding the SALCs:")
    print(f"> There are {len(structure.types)} atoms in the primitive cell")

    for i, nncluster in enumerate(structure.nnclusters):
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        print(f"> Solve vectorspace for atom {i+1} ")
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        print(" > Linear combinations")
        print_namedLCs(named_lcs)
        print(" ")
