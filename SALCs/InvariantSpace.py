from e3nn.o3 import Irreps
import torch
from typing import List, Union, Any
import math
# AO representation

from symmetry import Symmetry
from character_table import D3h_table
from torch_type import float_type

def apply_symmetry_operation(
        sym: Symmetry, irreps: Union[str, Irreps], values: torch.tensor
    ) -> torch.tensor:
    repres = Irreps(irreps)
    assert repres.dim == values.shape[-1]
    rotation = sym.get_matrix(repres)
    return torch.einsum("ij,kj -> ki", rotation, values)

# AO representation
class AORepresentation:
    """
    it will be a subspace spaned by all the given basis vector, plus the group operations
    """
    defined_orbitals = "s x z y xy xz z2 zy x2-y2".split()
    permute_yz = torch.tensor([0, 2, 1])
    def __init__(self, positions: Any, AOs: List[Union[Irreps, str]], symmetries: List[Symmetry]) -> None:
        positions = torch.tensor(positions, dtype = float_type)
        self.positions = torch.index_select(positions, 1, self.permute_yz)
        self.symmetry_operations = symmetries
        self.check_symmetry()

        self.nsym = len(self.symmetry_operations)
        self.natom = len(self.positions)
        assert len(AOs) == self.natom
        self.AOs = [Irreps(ir) for ir in AOs]
        self.bfs = [(iatom,j) for iatom in range(self.natom) for j in range(self.AOs[iatom].dim)]
        self.bf_index = { bf:i for i, bf in enumerate(self.bfs)}
        self.nbfs = len(self.bfs)
        self.reducible_matrix = self.generate_representation() # [nsym, nbf, nbf]
        self.reducible_trace = [float(torch.trace(m)) for m in self.reducible_matrix]

    def generate_representation(self):
        matrix = torch.zeros((self.nsym, self.nbfs, self.nbfs), dtype = float_type)
        for isym,sym in enumerate(self.symmetry_operations):
            rotated_positions = apply_symmetry_operation(sym, "1x1o", self.positions)
            for iatom, ipos in enumerate(self.positions):
                for jatom, jpos in enumerate(rotated_positions):
                    if torch.allclose(ipos, jpos):
                        # iatom is rotated to jatom
                        assert self.AOs[iatom] == self.AOs[jatom]
                        rotated_AOs = apply_symmetry_operation(sym, self.AOs[iatom], torch.eye(self.AOs[iatom].dim, dtype = float_type))
                        for iao in range(self.AOs[iatom].dim):
                            for jao in range(self.AOs[iatom].dim):
                                i_index = self.bf_index[(iatom,iao)]
                                j_index = self.bf_index[(jatom,jao)]
                                matrix[isym, j_index, i_index] = rotated_AOs[iao, jao] # equation 4.5
        return matrix
                        

    def check_symmetry(self):
        # check if symmetry is indeed the same
        #cartesian_positions = np.einsum("ij, kj -> ki", self.cpt[0],self.cpt[1])
        # we use strings to compare positions
        cartesian_positions = self.positions
        pos_list = set(["{:>7.3f},{:>7.3f},{:>7.3f}".format(*(p+0.000001)) for p in cartesian_positions.detach()])
        for symmetry in self.symmetry_operations:
            transformed = apply_symmetry_operation(symmetry, "1x1o", cartesian_positions)
            transformed = set(["{:>7.3f},{:>7.3f},{:>7.3f}".format(*(p+0.000001)) for p in transformed.detach()])
            assert transformed == pos_list, (transformed, symmetry)


D3h_positions = [
    [ 2.0,   0.0, 0.0] ,
    [-1.0, math.sqrt(3), 0.0],
    [-1.0,-math.sqrt(3), 0.0],
]
D3h_AOs = [ '1x1o' for pos in D3h_positions ]


D3h_AOs = AORepresentation(D3h_positions, D3h_AOs, D3h_table.symmetries)
