from e3nn.o3 import Irreps
import torch
from typing import List, Union, Any, Dict, Tuple
from symmetry import Symmetry
from atomic_orbital import LinearCombination
from character_table import GroupCharacterTable
from torch_type import float_type

def apply_symmetry_operation(
        sym: Symmetry, irreps: Union[str, Irreps], values: torch.tensor
    ) -> torch.tensor:
    repres = Irreps(irreps)
    assert repres.dim == values.shape[-1]
    rotation = sym.get_matrix(repres)
    return torch.einsum("ij,kj -> ki", rotation, values)

# AO representation
class VectorSpace:
    """
    it will be a subspace spaned by all the given basis vector, as well as the group operations,
    each linear combination serve as an initial basis function
    """
    def __init__(self, LCs: List[LinearCombination], symmetries: List[Symmetry]) -> None:
        self.LCs = LCs
        self.symmetries = symmetries
        self.check_symmetry()

        self.all_coefficients = torch.vstack([lc.coefficient for lc in self.LCs])
        self.rank = torch.linalg.matrix_rank(self.all_coefficients)

        self.has_representation = False
        self._reducible_matrix = None
        
    @property
    def reducible_matrix(self):
        if not self.has_representation:
            self._reducible_matrix = self.generate_representation() # [nsym, nbf, nbf]
            self.has_representation = True
        return self._reducible_matrix

    @property
    def reducible_trace(self):
        return [float(torch.trace(m)) for m in self.reducible_matrix]

    def _transformation_relationship(self):
        # this is not used
        # we determine how AOs are transfromed: | jatom, jorbit > = P | iatom, iorbit > 
        lc = self.LCs[0]
        nsym_class = len(self.symmetries)
        naos = len(lc.coefficient)
        transformation = torch.zeros((nsym_class, naos, naos), dtype = float_type)
        # < i,ao1 | P | j, ao2 >
        for isym, sym in enumerate(self.symmetries):
            for j, jatom in enumerate(lc.AOs):
                rotated_atom, rotated = jatom.rotate(sym)
                for i, iatom in enumerate(lc.AOs):
                    if rotated_atom == iatom:
                        assert iatom.irreps == jatom.irreps
                        for iorbit in range(iatom.irreps.dim):
                            for jorbit in range(jatom.irreps.dim):
                                index_i = lc.AO_index[(i,iorbit)]
                                index_j = lc.AO_index[(j,jorbit)]
                                transformation[isym, index_i, index_j] = rotated[jorbit, iorbit]
                        break
        return transformation

    def generate_representation(self):
        # determine how LCs are transformed
        # we do not assume coefficients are diagonal Identity matrix, 
        # we operate on the matrix and find the inner product
        # D_{ij} = < iLC | P | jLC >
        nsym_class = len(self.symmetries)
        nbfs = len(self.LCs)
        matrix = torch.zeros((nsym_class, nbfs, nbfs), dtype = float_type)
        for isym, sym in enumerate(self.symmetries):
            for j, jLC in enumerate(self.LCs):
                p_jLC = jLC.get_rotated(sym)
                for i, iLC in enumerate(self.LCs):
                    # take dot product
                    sum = 0
                    for iao1, ao1 in enumerate(p_jLC.AOs):
                        for iao2, ao2 in enumerate(iLC.AOs):
                            if ao1 == ao2:
                                for ir in range(ao1.irreps.dim):
                                    iLC_idx = iLC.AO_index[(iao2, ir)]
                                    p_jLC_idx = iLC.AO_index[(iao1, ir)]
                                    sum += iLC.coefficient[iLC_idx] * p_jLC.coefficient[p_jLC_idx]
                                break
                    matrix[isym, i, j] = sum
        return matrix

    def check_symmetry(self):
        # we only need to use one linear combination
        # we use strings to compare positions
        lc0 = self.LCs[0]
        cartesian_positions = torch.vstack([ ao.position for ao in lc0.AOs ])
        pos_list = set(["{:>7.3f},{:>7.3f},{:>7.3f}".format(*(p+0.000001)) for p in cartesian_positions.detach()])
        for symmetry in self.symmetries:
            transformed = apply_symmetry_operation(symmetry, "1x1o", cartesian_positions)
            transformed = set(["{:>7.3f},{:>7.3f},{:>7.3f}".format(*(p+0.000001)) for p in transformed.detach()])
            assert transformed == pos_list, (transformed, symmetry)


def decompose_vectorspace(
    reducible: VectorSpace, 
    table: GroupCharacterTable, 
    all_sym: List[torch.Tensor],
    normalize: bool = False
) -> Tuple[List[str], LinearCombination]:
    rep_names = [ irrep.symbol for irrep in table.irreps ]
    # we do under subspaces are one dimension or zero dimension
    subspaces


def decompose_vectorspace2(
    reducible: VectorSpace, 
    table: GroupCharacterTable, 
    all_sym: List[torch.Tensor],
    normalize: bool = False
) -> Dict[str,VectorSpace]:
    rep_names = [ irrep.symbol for irrep in table.irreps ]
    #coefficient = table.decompose_representation(reducible.reducible_trace)
    subspaces = {}
    for irrep_name in rep_names:
        fs = []
        characters = table.irreps_characters[irrep_name]
        dimension = characters[0] # dimension of irrep
        for lc in reducible.LCs:
            new_LC = torch.zeros_like(lc.coefficient, dtype=float_type)
            for isym, _ in enumerate(reducible.reducible_matrix):
                for full_matrix in all_sym[isym]:
                    new_LC += torch.matmul(full_matrix, lc.coefficient) * characters[isym] * dimension / table.order
            fs.append(LinearCombination(new_LC, lc.AOs, normalize))  
        subspaces[irrep_name] = VectorSpace(fs, reducible.symmetries)
    return subspaces

