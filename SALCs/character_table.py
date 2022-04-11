from dataclasses import dataclass
from typing import List
import torch
from symmetry import Symmetry

@dataclass
class IrreducibleRP:
    symbol: str
    characters: List[float]


class GroupCharacterTable:
    def __init__(self, 
        symmetries: List[Symmetry],
        class_multiplicity: List[int],
        irreps: List[IrreducibleRP]
    ) -> None:
        # information about the group
        self.symmetries = symmetries                          
        self.class_multiplicity = class_multiplicity   
        self.nclass = len(class_multiplicity) 
        self.irreps = irreps 
        self.order = sum(class_multiplicity)  # order of the group
        self.irreps_characters = { irrep.symbol:irrep.characters for irrep in self.irreps }
        for irrep in irreps:
            assert self.nclass == len(irrep.characters)

    def decompose_representation(self, reducible_trace: List[int]) -> dict:
        """
        returns the coefficients a_i which decompose reducible representation to irreducible ones
        """
        coefficients = {}
        assert len(reducible_trace) == self.nclass
        for irreducible in self.irreps:
            summation = 0
            for (ci, cr, nk) in zip(irreducible.characters, reducible_trace, self.class_multiplicity):
                summation += ci * cr * nk
            coeff = summation / self.order
            assert coeff % 1 < 1e-3, coeff  # should be integer
            coefficients[irreducible.symbol] = int(coeff)
        return coefficients

