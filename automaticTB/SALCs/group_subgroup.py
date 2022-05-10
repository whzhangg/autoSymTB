#
# we define all groups as a subgroup of one parent group. For each group, we select one
# unique subgroup and supergroup (maybe subgroups are the only one used). 
# we do not need to care about the detail of the symmetry 
# operation

# should we define groups (character table) as a datastructure, or an object? 
from group_data import groupData, IrreducibleRP, Possible_symmetry
from typing import List
from symmetry import Symmetry

class GroupTable:
    def __init__(self, name:str) -> None:
        if name not in groupData.keys():
            raise KeyError(f"{name} is not a defined point group symbol")
        self.data = groupData[name]

    @property
    def Schoenflies(self) -> str:
        return self.data.name_Schoenflies

    @property
    def HM_short(self) -> str:
        return self.data.name_HM_short

    @property
    def order(self) -> str:
        return self.data.order

    @property 
    def num_rep(self) -> int:
        return len(self.rep_symbol)

    @property
    def rep_symbol(self) -> List[str]:
        return [ rep.symbol for rep in self.data.Irreps ]

    @property
    def irreps(self) -> List[IrreducibleRP]:
        return self.data.Irreps

    @property
    def symmetries(self) -> List[Symmetry]:
        return [ Possible_symmetry[i] for i in self.data.symmetry_index]

    def __repr__(self) -> str:
        return f"{self.Schoenflies} with {self.order} Symmetry and {self.num_rep} Representations, subgroup: {self.data.subgroup_Schoenflies}"
    
    def get_subgroup(self):
        # return a subgroup
        if self.data.subgroup_Schoenflies == None:
            return None
        else:
            return GroupTable(self.data.subgroup_Schoenflies)

    # we do not expose character table.
    def decompose_representation(self, reducible_trace: List[int]) -> dict:
        """
        returns the coefficients a_i which decompose reducible representation to irreducible ones
        """
        coefficients = {}
        assert len(reducible_trace) == self.order
        for irreducible in self.data.Irreps:
            summation = 0
            for (ci, cr) in zip(irreducible.characters, reducible_trace):
                summation += ci * cr
            coeff = summation / self.order
            assert coeff % 1 < 1e-3, coeff  # should be integer
            coefficients[irreducible.symbol] = int(coeff)
        return coefficients

if __name__ == "__main__":
    D3h = GroupTable("D3h")
    Cs = D3h.get_subgroup()
    print(D3h)
    print(Cs)
