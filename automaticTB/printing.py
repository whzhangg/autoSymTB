import numpy as np
import typing
from .tools import Pair
from .atomic_orbitals import AO
from .SALCs import NamedLC
from .structure import NearestNeighborCluster
from .interaction import InteractionPairs

_print_lm = {
    (0,0): " s",
    (1,-1): "py",
    (1,0): "pz",
    (1,1): "px",
}

def _parse_orbital(l: int, m: int) -> str:
    if (l,m) not in _print_lm:
        return f"l={l};m={m}"
    else:
        return _print_lm[(l,m)]

def print_namedLCs(namedLcs: typing.List[NamedLC]):
    for nlc in namedLcs:
        print(nlc.name)
        print(nlc.lc)

def print_ao_pairs(nncluster: NearestNeighborCluster, pairs: typing.List[Pair]):
    print("Free interaction parameters: ")
    for i, pair in enumerate(pairs):
        left: AO = pair.left
        right: AO = pair.right
        result = "{:>3d}".format(i+1) + " > Free AO interaction: "
        rij = nncluster.baresites[right.cluster_index].pos - nncluster.baresites[left.cluster_index].pos
        result += f"{left.chemical_symbol} {_parse_orbital(left.l, left.m)} -> "
        result += f"{right.chemical_symbol} {_parse_orbital(right.l, right.m)} @ "
        result += "({:>6.2f},{:>6.2f},{:>6.2f})".format(*rij)
        print(result)

    
def print_InteractionPairs(nncluster: NearestNeighborCluster, interactionpairs: InteractionPairs):
    print("Solved Interactions: ")
    for i, pair_value in enumerate(interactionpairs.AO_energies):
        left: AO = pair_value.left
        right: AO = pair_value.right
        result = "{:>3d}".format(i+1) + " > AO interaction: "
        rij = nncluster.baresites[right.cluster_index].pos - nncluster.baresites[left.cluster_index].pos
        result += f"{left.chemical_symbol} {_parse_orbital(left.l, left.m)} -> "
        result += f"{right.chemical_symbol} {_parse_orbital(right.l, right.m)} @ "
        result += "({:>6.2f},{:>6.2f},{:>6.2f})".format(*rij)
        result += "H: {:>6.2f}".format(pair_value.value)
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