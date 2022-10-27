# this script provide high level function to obtain a structure from given cif file

from automaticTB.structure import Structure
from automaticTB.tools import read_cif_to_cpt
import re, typing

__all__ = ["get_structure_from_cif_file"]

def get_structure_from_cif_file(
    filename: str, orbitals_dict: typing.Dict[str, str]) -> Structure:
    """
    use voronoi method to find the coordinated atoms for interaction, since it is parameter free
    - filename: the cif file
    - orbitals_dict: {"Si" : "3s3p4s"}
    """
    c, p, t = read_cif_to_cpt(filename)
    orbitals = {
        k: _get_orbital_ln_from_string(v) for k, v in orbitals_dict.items()
    }

    return Structure.from_cpt_voronoi(c, p, t, orbitals)


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