from ase import io, Atoms
import numpy as np, typing
from ...parameters import precision_decimal


def atom_from_cpt_fractional(lattice: np.ndarray, positions: np.ndarray, types) -> Atoms:
    result = Atoms(cell = lattice, scaled_positions = positions)
    if type(types[0]) == str:
        result.set_chemical_symbols(types)
    else:
        result.set_atomic_numbers(types)
    return result

def atom_from_cpt_cartesian(lattice: np.ndarray, cart_positions: np.ndarray, types) -> Atoms:
    invcell = np.linalg.inv(lattice.T)
    frac = np.einsum("ij, kj -> ki", invcell, cart_positions)
    return atom_from_cpt_fractional(lattice, frac, types)


def write_cif_file(filename: str,struct: Atoms):
    io.write(filename, struct, format='cif')
