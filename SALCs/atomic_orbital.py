from e3nn.o3 import Irreps
from typing import Union, Tuple, List
from symmetry import Symmetry
import torch 
from torch_type import float_type

AO_SYMBOL = {
    "0e": ["s"],
    "1o": ["px","pz","py"],
    "2e": "xy xz z2 zy x2-y2".split()
}

class CenteredAO:
    """
    AO centered on atoms at a given position, we will have one object 
    on each class, the important data in this class is the position,
    the coefficients are detached from this object. They will be fixed on each atoms,
    we take note that input position is x, z, y, because of the definition of rotation
    """
    def __init__(self, atomic_symbol: str, position: torch.Tensor, irreps: Union[str, Irreps]) -> None:
        assert len(position) == 3
        self.position = position
        self.irreps = Irreps(irreps)
        self.atom_symbol = atomic_symbol
        self.ao_symbol = []
        for multi, irrep in self.irreps:
            assert multi == 1,  "For now, we assume only one orbital per function"
            self.ao_symbol += AO_SYMBOL[str(irrep)]
        
    def rotate(self, sym: Symmetry) -> Tuple[torch.Tensor]:
        """
        return the rotated 
        """
        rotation_cartesian = sym.get_matrix("1x1o")
        pos2 = torch.einsum("ij, j -> i", rotation_cartesian, self.position)
        orbital_coefficient = sym.get_matrix(self.irreps).T
        return CenteredAO(self.atom_symbol, pos2, self.irreps), orbital_coefficient

    def __eq__(self, other) -> bool:
        if torch.allclose(self.position, other.position) and self.irreps == other.irreps:
            return True
        else: 
            return False


class LinearCombination:
    def __init__(self, coefficient: torch.Tensor, AOs: List[CenteredAO]) -> None:
        assert coefficient.shape[0] == sum([ao.irreps.dim for ao in AOs])
        self.coefficient = coefficient
        self.AOs = AOs
        self.AO_name = [ (iao,i) for iao, ao in enumerate(self.AOs) for i in range(ao.irreps.dim)]
        self.AO_index = { ao:i for i, ao in enumerate(self.AO_name)}

    def normalized(self):
        # return a normalized version of the same vector
        n = torch.norm(self.coefficient)
        if n < 1e-5:
            normalized = self.coefficient
        else:
            normalized = self.coefficient / torch.norm(self.coefficient)
        return LinearCombination(normalized, self.AOs)

    def __str__(self, tol = 1e-5):
        sign = {
            1: "+",
            -1: "-"
        }
        result = ""
        ao_names = [ f"{ao.atom_symbol}-{aos}" for ao in self.AOs for aos in ao.ao_symbol]
        for s, absc, ao in zip(torch.sign(self.coefficient), torch.abs(self.coefficient), ao_names):
            if absc < tol: continue
            result += f" {sign[s.item()]} " + "{:>5.3f}".format(absc.item()) + f" {ao}"
        return result

    def get_rotated(self, sym: Symmetry):
        # return another LinearCombination,
        new_coefficient = []
        new_AOlist = []
        for i, iao in enumerate(self.AOs):
            new_ao, rotated_coeff = iao.rotate(sym)
            new_AOlist.append(new_ao)
            for iorbit in range(iao.irreps.dim):
                rotated_coeff[iorbit, :] *= self.coefficient[self.AO_index[(i, iorbit)]]
            new_coefficient.append(torch.sum(rotated_coeff, dim=0))
        new_coefficient = torch.hstack(new_coefficient)

        return LinearCombination(new_coefficient, new_AOlist)
        
    def plot(self):
        raise NotImplementedError

def get_dimension(AOs: List[CenteredAO]) -> int:
    return sum([ao.irreps.dim for ao in AOs])

def get_linear_combination(matrix: torch.Tensor, AOs: List[CenteredAO]) -> List[LinearCombination]:
    result = [ LinearCombination(row, AOs) for row in matrix]
    return result


if __name__ == "__main__":
    ao1 = CenteredAO("atom1", torch.tensor([0.0, 0.0, 0.0], dtype = float_type), "1o")
    ao2 = CenteredAO("atom2", torch.tensor([0.5, 0.0, 0.0], dtype = float_type), "1o")
    lc = LinearCombination(torch.tensor([0.0, 0.1, 0.1, 0.8, -0.2, 0.05], dtype = float_type), [ao1, ao2])
    print(lc.AO_name, lc.AO_index)
    print(lc)
    # >>> + 0.100 atom1-pz + 0.100 atom1-py + 0.800 atom2-px - 0.200 atom2-pz + 0.050 atom2-py