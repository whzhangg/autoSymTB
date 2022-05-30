from ..atomic_orbitals import AO
from ..utilities import Pair
import typing

def print_ao_pairs(aos: typing.List[AO], pairs: typing.List[Pair]):
    print("Free interaction parameters: ")
    for i, pair in enumerate(pairs):
        leftIndex = pair.left
        rightIndex = pair.right
        left: AO = aos[leftIndex]
        right: AO = aos[rightIndex]
        print("{:>3d}".format(i+1)+f" > Free AO interaction: {str(left)} <-> {str(right)}")