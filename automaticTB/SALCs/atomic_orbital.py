from e3nn.o3 import Irreps
from typing import Union, Tuple, List
import torch 
from torch_type import float_type
import numpy as np
from plot_mo import wavefunction, get_meshgrid

AO_SYMBOL = {
    "0e": ["s"],
    "1o": ["px","pz","py"],
    "2e": "xy xz z2 zy x2-y2".split()
}

class CenteredAO:
    """
    A Datastructure that can compare and store information about its name
    """
    def __init__(self, atomic_symbol: str, position: torch.Tensor, irreps: Union[str, Irreps]) -> None:
        assert len(position) == 3
        self.position = position
        self.irreps = Irreps(irreps)
        self.atom_symbol = atomic_symbol

        for multi, irrep in self.irreps:
            assert multi == 1,  "For now, we assume only one orbital per function"

    @property
    def ao_symbol(self):
        return [ f"{self.atom_symbol}-{orb_sym}" for _, irrep in self.irreps for orb_sym in AO_SYMBOL[str(irrep)] ]
        
    @property
    def orbit_symbol(self):
        return [ orb_sym for _, irrep in self.irreps for orb_sym in AO_SYMBOL[str(irrep)] ]
        
    def __eq__(self, other) -> bool:
        if torch.allclose(self.position, other.position) and self.irreps == other.irreps:
            return True
        else: 
            return False

    def __repr__(self) -> str:
        results = str(self.irreps)
        return "{:s} with {:s} at ({:>7.3f},{:>7.3f},{:>7.3f})".format(self.atom_symbol, results, *self.position)

class LinearCombination:
    display_tol = 1e-5
    def __init__(self, coefficient: torch.Tensor, AOs: List[CenteredAO], normalize: bool = False) -> None:
        assert coefficient.shape[0] == sum([ao.irreps.dim for ao in AOs])
        if normalize:
            self.coefficient = self._normalize_coefficient(coefficient)
        else:
            self.coefficient = coefficient
        
        self.AOs = AOs
        self._AO_name = [ (iao,i) for iao, ao in enumerate(self.AOs) for i in range(ao.irreps.dim)]
        self._AO_index = { ao:i for i, ao in enumerate(self._AO_name)}

    def get_positions(self) -> torch.Tensor:
        return torch.vstack([ ao.position for ao in self.AOs ])

    def get_AO_index(self, iAO:int, iorb:int) -> int:
        return self._AO_index[(iAO, iorb)]

    def normalized(self):
        # return a normalized version of the same vector
        normalized = self._normalize_coefficient(self.coefficient)
        return LinearCombination(normalized, self.AOs)

    def _normalize_coefficient(self, coefficient: torch.Tensor):
        n = torch.norm(coefficient)
        if n < 1e-5:
            return coefficient
        else:
            return coefficient / n

    def __str__(self):
        sign = {
            1: "+",
            -1: "-"
        }
        result = ""
        ao_names = []
        for ao in self.AOs:
            ao_names += ao.ao_symbol
        for s, absc, ao in zip(torch.sign(self.coefficient), torch.abs(self.coefficient), ao_names):
            if absc < self.display_tol: continue
            result += f" {sign[s.item()]} " + "{:>5.3f}".format(absc.item()) + f" {ao}"
        return result
        
    def __bool__(self):
        return torch.norm(self.coefficient).item() > self.display_tol

    def density(self):
        grid_size = 50
        xs, ys, zs = get_meshgrid(self.get_positions().numpy(), grid_size)
        result = np.zeros_like(xs)
        for iao, (iatom, iorbit) in enumerate(self._AO_name):
            co = self.coefficient[iao]
            position_moved = torch.index_select(self.AOs[iatom].position, 0, torch.tensor([0,2,1]))
            position = position_moved.numpy()
            name = self.AOs[iatom].orbit_symbol[iorbit]
            result += co.item() * wavefunction(1, name, position, xs, ys, zs)
        return xs, ys, zs, result
            

class VectorSpace:
    """
    it will be a subspace spaned by all the given basis vector, as well as the group operations,
    each linear combination serve as an initial basis function
    """
    def __init__(self, LCs: List[LinearCombination]) -> None:
        self.LCs = LCs
        self._all_coefficients = torch.vstack([lc.coefficient for lc in self.LCs])

    @property
    def rank(self):
        return torch.linalg.matrix_rank(self._all_coefficients, tol = 1e-5 )

    def __repr__(self) -> str:
        return f"VectorSpace with rank {self.rank} and {len(self.LCs)} LCs"

    def remove_linear_dependent(self):
        non_zero = [lc for lc in self.LCs if lc]
        independent = [non_zero[0]]
        for vector in non_zero:
            is_close = False
            for compare in independent:
                cos = torch.abs(torch.dot(vector.coefficient, compare.coefficient))
                if torch.isclose(cos, torch.tensor(1, dtype = float_type)):
                    is_close = True
                    break
            if not is_close and vector:

                independent.append(vector)
        return VectorSpace(independent)

    @property
    def display_str(self) -> str:

        any_function = any([bool(lc) for lc in self.LCs])
        if self.rank == 0 or not any_function:
            return ""

        result = f"Vector space with rank {self.rank}"
        for lc in self.LCs:
            if lc: result += f"\n{str(lc)}"
        return result

class VectorSpacewithBasis(VectorSpace):
    # it is basically a VectorSpace with named Linear combinations are basis
    def __init__(self, LCs: List[LinearCombination], names: List[str] ) -> None:
        assert len(LCs) == len(names)
        super().__init__(LCs)
        self.names = names

    @property
    def display_str(self) -> str:

        any_function = any([bool(lc) for lc in self.LCs])
        if self.rank == 0 or not any_function:
            return ""

        result = f"Vector space with rank {self.rank}"
        for lc, name in zip(self.LCs, self.names):
            if lc: result += f"\n{name}: {str(lc)}"
        return result


def get_coefficient_dimension(AOs: List[CenteredAO]) -> int:
    return sum([ao.irreps.dim for ao in AOs])


def get_linear_combination(matrix: torch.Tensor, AOs: List[CenteredAO]) -> List[LinearCombination]:
    result = [ LinearCombination(row, AOs) for row in matrix]
    return result


if __name__ == "__main__":
    ao1 = CenteredAO("atom1", torch.tensor([0.0, 0.0, 0.0], dtype = float_type), "1o")
    ao2 = CenteredAO("atom2", torch.tensor([3, 0.0, 0.0], dtype = float_type), "1o")
    lc1 = LinearCombination(torch.tensor([0.0, 1.0, 0.0, 0.0,  0.0, 0.0], dtype = float_type), [ao1, ao2])
    lc2 = LinearCombination(torch.tensor([0.0, 0.0, 0.0, 0.0, -0.0, 0.0], dtype = float_type), [ao1, ao2])
    vs = VectorSpace([lc1, lc2])

    print(ao1)
    print(lc1)
    print(vs)
    print(vs.display_str)

    from plot_mo import get_meshgrid, plot_density
    positions = lc1.get_positions()
    xs, ys, zs = get_meshgrid(positions.numpy())
    
    density = lc1.density([xs, ys, zs])
    plot_density(xs, ys, zs, density, "lc.html")
    # >>> + 0.100 atom1-pz + 0.100 atom1-py + 0.800 atom2-px - 0.200 atom2-pz + 0.050 atom2-py