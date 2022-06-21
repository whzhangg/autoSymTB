import numpy as np
import os, typing
from automaticTB.structure import NearestNeighborCluster, Structure
from automaticTB.examples.Perovskite.structure import get_perovskite_structure
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
    struct = atom_from_cpt(struct.cell, struct.positions, struct.types)
    write_cif_file(filename, struct)


def write_structure():
    structure = get_perovskite_structure()
    write_cif_structure(structure, "perovskite.cif")
    for i, nncluster in enumerate(structure.nnclusters):
        write_cif_NNcluster(nncluster, f"cluster{i+1}.cif")


if __name__ == "__main__":
    write_structure()