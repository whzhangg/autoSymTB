import typing
import numpy as np
from ..tools import Pair, tensor_dot, find_free_variable_indices
from ..tools.mathLA import row_echelon
from ..structure import NearestNeighborCluster, ClusterSubSpace
from ..parameters import zero_tolerance
from ..SALCs import NamedLC, IrrepSymbol

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
    for pair in nncluster.interaction_subspace_pairs:
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


class HomogeneousEquationFinder:
    def __init__(self) -> None:
        self.memory = {}

    
    def get_equation(self, rep1: IrrepSymbol, rep2: IrrepSymbol, tp: np.ndarray) -> typing.Optional[np.ndarray]:
        if rep1.symmetry_symbol != rep2.symmetry_symbol:
            return tp

        main1 = f"{rep1.main_irrep}^{rep1.main_index}"
        main2 = f"{rep2.main_irrep}^{rep2.main_index}"
        pair = " ".join([main1, main2])
        if pair not in self.memory:
            self.memory[pair] = tp
            return None
        else:
            return tp - self.memory[pair]


def get_free_interaction_AO(nncluster: NearestNeighborCluster, named_lcs: typing.List[NamedLC], debug = False) \
-> typing.List[Pair]:
    transformer = get_block_transformer(nncluster, named_lcs)
    ao_pairs = []

    for conversion in transformer:
        finder = HomogeneousEquationFinder()
        equations = []
        assert len(conversion.matrix) == len(conversion.mo_pairs)
        for row, mopair in zip(conversion.matrix, conversion.mo_pairs):
            left_rep = named_lcs[mopair.left].name
            right_rep = named_lcs[mopair.right].name
            if debug:
                print(" - ".join([str(left_rep), str(right_rep)]))
            found = finder.get_equation(left_rep, right_rep, row)
            if not (found is None):
                equations.append(found)
                #if debug:
                #    print_matrix(np.array([found.real]))
        
        if len(equations) == 0:
            ao_pairs += conversion.ao_pairs
            continue

        homogeneous_equations = np.array(equations)
        free_indices = find_free_variable_indices(homogeneous_equations.real)
        ao_pairs += [conversion.ao_pairs[i] for i in free_indices]

        if debug:
            #print(homogeneous_equations.real)
            #print_matrix(row_echelon(homogeneous_equations.real))
            print(free_indices)
            #for pair in [conversion.ao_pairs[i] for i in free_indices]:
                #print_ao_pairs(nncluster, [pair])

    return ao_pairs
