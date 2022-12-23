import numpy as np, typing
from ...tools import atom_from_cpt

def make_supercell_using_ase(
    cpt: typing.Tuple[np.ndarray, np.ndarray, np.ndarray],
    supercell_matrix: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ase.build import make_supercell
    primitive = atom_from_cpt(*cpt)
    supercell = make_supercell(primitive, supercell_matrix)
    return (
        supercell.get_cell(),
        supercell.get_scaled_positions(),
        supercell.get_atomic_numbers()
    )


