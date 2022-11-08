from ase import io, Atoms
import numpy as np, typing


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


def get_absolute_pos_from_cell_fractional_pos_translation(
    cell: np.ndarray, x: np.ndarray, t: np.ndarray
) -> np.ndarray:
    return cell.T.dot(x + t)


def get_home_position_and_cell_translation(fractional_positions: np.ndarray) \
-> typing.Tuple[np.ndarray, np.ndarray]:
    """
    we first determine the position in the primitive cell, 
    """
    translation = np.floor(fractional_positions)
    home = fractional_positions - translation
    home[np.where(np.isclose(home, np.floor(home), atol=1e-10))] = 0.0
    translation = fractional_positions - home
    return home, translation


def position_string(pos: np.ndarray) -> str:
    """
    convert position to string to compare positions
    """
    return "{:>+12.6f}{:>+12.6f}{:>+12.6f}".format(pos[0], pos[1], pos[2])
