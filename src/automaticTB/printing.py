import numpy as np
import typing
from .tools import Pair
from .atomic_orbitals import AO
from .SALCs import NamedLC
from .structure import Structure
from .interaction import AOPairWithValue

__all__ = [
    "get_orbital_symbol_from_lm", # used in bandstructure plots
    "print_namedLCs", 
    "print_matrix",
    "print_ao_pairs",
    "print_InteractionPairs"
]

_print_lm = {
    (0, 0): "s",
    (1,-1): "py",
    (1, 0): "pz",
    (1, 1): "px",
    (2,-2): "dxy",
    (2,-1): "dyz",
    (2, 0): "dz2",
    (2, 1): "dxz",
    (2, 2): "dx2-y2"
}

def get_orbital_symbol_from_lm(l: int, m: int) -> str:
    if (l,m) not in _print_lm:
        raise "l,m not supported"
    return _print_lm[(l,m)]


def _parse_orbital(n: int, l: int, m: int) -> str:
    if (l,m) not in _print_lm:
        return f"n={n};l={l};m={m}"
    else:
        formatted = _print_lm[(l,m)]
        return f"{n}{formatted}"

def print_namedLCs(namedLcs: typing.List[NamedLC]):
    for nlc in namedLcs:
        print(nlc.name)
        print(nlc.lc)

def print_ao_pairs(structure: Structure, pairs: typing.List[Pair]):
    for i, pair in enumerate(pairs):
        left: AO = pair.left
        right: AO = pair.right
        result = "{:>3d}".format(i+1) + " > Pair: "
        lpos = structure.cell.T.dot(
            left.translation + structure.positions[left.primitive_index]
        )
        rpos = structure.cell.T.dot(
            right.translation + structure.positions[right.primitive_index]
        )
        rij = rpos - lpos
        result += f"{left.chemical_symbol:>2s}-{left.primitive_index:0>2d} {_parse_orbital(left.n, left.l, left.m):>7s} -> "
        result += f"{right.chemical_symbol:>2s}-{right.primitive_index:0>2d}{_parse_orbital(right.n, right.l, right.m):>7s} @ "
        result += "({:>6.2f},{:>6.2f},{:>6.2f})".format(*rij)
        print(result)

    
def print_InteractionPairs(
    structure: Structure, interactionpairs: typing.List[AOPairWithValue]
):
    print("Solved Interactions: ")
    for i, pair_value in enumerate(interactionpairs):
        left: AO = pair_value.left
        right: AO = pair_value.right
        result = "{:>3d}".format(i+1) + " > AO interaction: "

        lpos = structure.cell.T.dot(
            left.translation + structure.positions[left.primitive_index]
        )
        rpos = structure.cell.T.dot(
            right.translation + structure.positions[right.primitive_index]
        )
        rij = rpos - lpos
        
        result += f"{left.primitive_index:>2d}{left.chemical_symbol:>2s}"
        result += f"{_parse_orbital(left.n, left.l, left.m):>7s} -> "
        result += f"{right.primitive_index:>2d}{right.chemical_symbol:>2s}"
        result += f"{_parse_orbital(right.n, right.l, right.m):>7s} @ "
        result += "({:>6.2f},{:>6.2f},{:>6.2f})".format(*rij)
        result += "H: {:>6.2f}".format(pair_value.value.real)
        print(result)


def print_matrix(m: np.ndarray, format:str = "{:>6.2f}"):
    # print an matrix using the given string format
    assert len(m.shape) == 2
    result = ""
    for row in m:
        for cell in row:
            result += " " + format.format(cell)
        result += "\n"
    print(result)