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


# a test data, group D3h
# http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=603&option=4


D3h_rotation = [
    Symmetry((0,       0,            0), 0), #0 E        identity
    Symmetry((0,       0,     torch.pi), 1), #1 Sigma_h  horizontal refection
    Symmetry((0,       0, 2*torch.pi/3), 0), #2 C3       120degree rotation
    Symmetry((0,       0,   torch.pi/3), 1), #3 S3       
    Symmetry((0,torch.pi,            0), 0), #4 C2'      rotation around x 
    Symmetry((0,torch.pi,    -torch.pi), 1)  #5 Sigma_v  vertical reflection
]

D3h_class_multiplicity = [1, 1, 2, 2, 3, 3]
D3h_irrepresentations = [
    IrreducibleRP("A1\'", [1, 1, 1, 1, 1, 1]),
    IrreducibleRP("A2\'", [1, 1, 1, 1,-1,-1]),
    IrreducibleRP("A1\"", [1,-1, 1,-1, 1,-1]),
    IrreducibleRP("A2\"", [1,-1, 1,-1,-1, 1]),
    IrreducibleRP("E\'",  [2, 2,-1,-1, 0, 0]),
    IrreducibleRP("E\"",  [2,-2,-1, 1, 0, 0])
]

D3h_table = GroupCharacterTable(
    D3h_rotation, D3h_class_multiplicity, D3h_irrepresentations
)
