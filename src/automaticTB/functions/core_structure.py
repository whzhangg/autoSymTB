# this script provide high level function to obtain a structure from given cif file

from automaticTB.structure import Structure
from automaticTB.tools import read_cif_to_cpt
import re, typing

__all__ = ["get_structure_from_cif_file"]


def get_structure_from_cif_file(
    filename: str, rcut: typing.Optional[float] = None
) -> Structure:
    """
    high level function to get a structure object from cif file
    """
    c, p, t = read_cif_to_cpt(filename)
    if rcut is None:
        return Structure.from_cpt_voronoi(c, p, t)
    else:
        return Structure.from_cpt_rcut(c, p, t, rcut)



def _get_orbital_ln_from_string(orbital: str) -> typing.List[typing.Tuple[int, int]]:
    """
    example "3s3p4s" -> [(3,0),(3,1),(4,0)]
    """
    l_dict = {
        "s": 0, "p": 1, "d": 2, "f": 3
    }
    numbers = re.findall(r"\d", orbital)
    symbols = re.findall(r"[spdf]", orbital)
    if len(numbers) != len(symbols):
        print(f"orbital formula wrong: {orbital}")
    return [
        (n,l_dict[lsym]) for n,lsym in zip(numbers, symbols)
    ]