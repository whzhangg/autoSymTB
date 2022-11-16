from automaticTB.solve.structure import Structure
from automaticTB.tools import read_cif_to_cpt
import typing

__all__ = ["get_structure_from_cif_file"]

def get_structure_from_cif_file(
    filename: str, 
    orbitals_dict: typing.Dict[str, str], 
    rcut: typing.Optional[float] = None,
    standardize: bool = True
) -> "Structure":
    """ high level function to get a structure object from cif file """
    c, p, t = read_cif_to_cpt(filename)
    return Structure(c, p, t, orbitals_dict, rcut, standardize)
