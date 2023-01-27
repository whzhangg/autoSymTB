"""This module defines interface to `ase` module"""

import os
import numpy as np

from ase import Atoms
from ase import io, data

atomic_numbers = data.atomic_numbers
chemical_symbols = data.chemical_symbols


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


def atom_from_cpt(lattice, positions, types) -> Atoms:
    result = Atoms(cell = lattice, scaled_positions = positions)
    if type(types[0]) == str:
        result.set_chemical_symbols(types)
    else:
        result.set_atomic_numbers(types)
    return result


def read_cif_to_cpt(filename: str) -> tuple:
    if not filename.endswith("cif"):
        raise RuntimeError(f"Structure file {filename} is not a .cif file !")
    elif not os.path.exists(filename):
        raise RuntimeError(f"Structure file {filename} cannot be found !")
    else:
        struct: Atoms = io.read(filename,format='cif')
        return struct.get_cell(), struct.get_scaled_positions(), struct.get_atomic_numbers()
