import numpy as np
import os, typing
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.structure import NearestNeighborCluster, Structure
from automaticTB.interaction import get_free_interaction_AO, InteractionEquation
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.Perovskite.structure import get_perovskite_structure
from automaticTB.printing import print_ao_pairs, print_matrix, print_InteractionPairs
from automaticTB.examples.Perovskite.ao_interaction import get_interaction_values_from_list_AOpairs
from automaticTB.plotting.adaptor import get_cell_from_origin_centered_positions

def write_cif_NNcluster(nncluster: NearestNeighborCluster, filename:str) -> None:
    types = [site.atomic_number for site in nncluster.baresites]
    positions = [site.pos for site in nncluster.baresites] 
    cell = get_cell_from_origin_centered_positions(positions)
    shift = np.dot(cell.T, np.array([0.5,0.5,0.5]))
    positions = positions + shift
    fractional = np.einsum('ij, kj -> ki', np.linalg.inv(cell.T), positions)
    from DFTtools.tools import atom_from_cpt, write_cif_file
    struct = atom_from_cpt(cell, fractional, types)
    write_cif_file(filename, struct)

def write_cif_structure(struct: Structure, filename:str) -> None:
    from DFTtools.tools import atom_from_cpt, write_cif_file
    struct = atom_from_cpt(struct.cell, struct.positions + np.array([0.5,0.5,0.5]), struct.types)
    write_cif_file(filename, struct)


def write_structure():
    structure = get_perovskite_structure()
    write_cif_structure(structure, "perovskite.cif")
    for i, nncluster in enumerate(structure.nnclusters):
        write_cif_NNcluster(nncluster, f"cluster{i+1}.cif")


# seems to be a bit problematic
def run_perovskite_SALCs():
    structure = get_perovskite_structure()
    print("Solving for the free nearest neighbor interaction in Perovskite")
    print("Starting ...")
    for i, nncluster in enumerate(structure.nnclusters):
        #if i == 0: continue
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        print("Solve Interaction ...")
        free_pairs = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs).free_AOpairs
        
        print_ao_pairs(nncluster, free_pairs)
        print()

def test_perovskite_SALCs():
    structure = get_perovskite_structure()
    print("Solving for the free nearest neighbor interaction in Perovskite")
    print("Starting ...")
    nnclusters = (structure.nnclusters[1],structure.nnclusters[2])
    for nncluster in nnclusters:
        #for seitz, op in nncluster.sitesymmetrygroup.dressed_op.items():
        #    print(seitz)
        #    print(print_matrix(op))
        #for irrep, value in nncluster.sitesymmetrygroup.irreps.items():
        #    print(irrep)
        #    print(value)
            
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        for nlcs in named_lcs:
            print(nlcs.name)
            print(nlcs.lc)
            print("")
        print("Solve Interaction ...")
        equation_system = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
        free_pairs = equation_system.free_AOpairs
        interactions = get_interaction_values_from_list_AOpairs(structure.cell, structure.positions, free_pairs)
        all_interactions = equation_system.solve_interactions_to_InteractionPairs(interactions)
        print_InteractionPairs(nncluster, all_interactions)
        print()

if __name__ == "__main__":
    write_structure()