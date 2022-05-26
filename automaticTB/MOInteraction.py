import typing
import numpy as np
from scipy.linalg import lu
from .utilities import Pair, tensor_dot
from .structure import NearestNeighborCluster, ClusterSubSpace
from .parameters import zero_tolerance
from .SALCs import NamedLC

class BlockTransformer:
    def __init__(
        self,
        left_ao_indices: typing.List[int],
        left_mo_indices: typing.List[int],
        right_ao_indices: typing.List[int],
        right_mo_indices: typing.List[int],
        coefficients: np.ndarray
    ):
        self._ao_indices_pairs = Pair(left_ao_indices, right_ao_indices)
        self._mo_indices_pairs = Pair(left_mo_indices, right_mo_indices)
        self.ao_pairs = tensor_dot(left_ao_indices, right_ao_indices)
        self.mo_pairs = tensor_dot(left_mo_indices, right_mo_indices)
        rows = []
        for mopair in self.mo_pairs:
            row = []
            for aopair in self.ao_pairs:
                row.append(
                    coefficients[mopair.left, aopair.left] \
                        * coefficients[mopair.right, aopair.right]
                )
            rows.append(row)
        self.matrix = np.array(rows)
    
    @property
    def is_selfinteraction(self) -> bool:
        return set(self._ao_indices_pairs.left) == set(self._ao_indices_pairs.right)

    @property
    def inv_matrix(self):
        return np.linalg.inv(self.matrix)
        

def get_block_transformer(nncluster: NearestNeighborCluster, named_lcs: typing.List[NamedLC]) \
-> typing.List[BlockTransformer]:
    stacked_coefficients = np.vstack([nlc.lc.coefficients for nlc in named_lcs])
    conversion_matrices: typing.List[BlockTransformer] = []
    for pair in nncluster.subspace_pairs:
        left: ClusterSubSpace = pair.left
        right: ClusterSubSpace = pair.right

        moindex_left = []
        moindex_right = []
        for i, namedlc in enumerate(named_lcs):
            if np.linalg.norm(namedlc.lc.coefficients[left.indices]) > zero_tolerance:
                # if this subspace is non-zero, then other coefficients would be necessary zero
                moindex_left.append(i)
            if np.linalg.norm(namedlc.lc.coefficients[right.indices]) > zero_tolerance:
                moindex_right.append(i)
        conversion_matrices.append(
            BlockTransformer(
                left.indices, moindex_left, right.indices, moindex_right, stacked_coefficients
            )
        )
    return conversion_matrices


class EquationFinder:
    def __init__(self) -> None:
        self.memory = {}

    
    def get_equation(self, rep1: str, rep2: str, tp: np.ndarray) -> typing.Optional[np.ndarray]:
        #tp = np.tensordot(namedLC1.lc.coefficients, namedLC2.lc.coefficients, axes=0)
        #rep1 = namedLC1.name
        #rep2 = namedLC2.name

        if rep1 != rep2:
            return tp

        # rep1 == rep2
        main1 = rep1.split("->")[0]
        main2 = rep2.split("->")[0]
        pair = " ".join([main1, main2])
        if pair not in self.memory:
            self.memory[pair] = tp
            return None
        else:
            return tp - self.memory[pair]

def find_free_variable_indices(A:np.ndarray) -> typing.Set[int]:
    P, L, U = lu(A)
    for row in U:
        print(row)
    basis_columns = {np.flatnonzero(U[i, :])[0] for i in range(U.shape[0])}
    free_variables = set(range(U.shape[1])) - basis_columns
    return free_variables

# We can also operate at the level of conversion matrix
def get_free_AO_pairs(nncluster: NearestNeighborCluster, named_lcs: typing.List[NamedLC]) \
-> typing.List[Pair]:
    stacked_coefficients = np.vstack([nlc.lc.coefficients for nlc in named_lcs])

    origin = nncluster.origin_index
    center_ao_indices = nncluster.orbitalslist.atomic_slice_dict[origin]

    ao_pairs = tensor_dot(nncluster.AOlist[center_ao_indices], nncluster.AOlist)

    homogeneous_equations = []
    for pair in nncluster.subspace_pairs:
        left: ClusterSubSpace = pair.left
        right: ClusterSubSpace = pair.right

        moindex_left: typing.List[int] = []
        moindex_right: typing.List[int] = []
        for i, namedlc in enumerate(named_lcs):
            if np.linalg.norm(namedlc.lc.coefficients[left.indices]) > zero_tolerance:
                # if this subspace is non-zero, then other coefficients would be necessary zero
                moindex_left.append(i)
            if np.linalg.norm(namedlc.lc.coefficients[right.indices]) > zero_tolerance:
                moindex_right.append(i)
        
        all_MO_pairs = tensor_dot(moindex_left, moindex_right)
        homogeneous_finder = EquationFinder()
        for mo_pair in all_MO_pairs:
            tp = np.tensordot(
                named_lcs[mo_pair.left].lc.coefficients[center_ao_indices],
                named_lcs[mo_pair.right].lc.coefficients, axes=0
                ).flatten()
            result = homogeneous_finder.get_equation(
                named_lcs[mo_pair.left].name, named_lcs[mo_pair.right].name, tp
            )
            if not (result is None): 
                homogeneous_equations.append(result)
    
    homogeneous_equations = np.array(homogeneous_equations)
    free_indices = find_free_variable_indices(homogeneous_equations.real)
    print(free_indices)
    return [ao_pairs[i] for i in free_indices]

