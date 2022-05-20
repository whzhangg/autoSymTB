import dataclasses, typing, collections
import numpy as np
from .utilities import Pair, tensor_dot
from .structure import NearestNeighborCluster, ClusterSubSpace
from .parameters import zero_tolerance
from .SALCs import NamedLC


class ConversionMatrix:
    def __init__(
        self,
        left_ao_indices: typing.List[int],
        left_mo_indices: typing.List[int],
        right_ao_indices: typing.List[int],
        right_mo_indices: typing.List[int],
        coefficients: np.ndarray
    ):
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
    def inv_matrix(self):
        return np.linalg.inv(self.matrix)
        

def get_conversion_matrices(nncluster: NearestNeighborCluster, named_lcs: typing.List[NamedLC]) \
-> typing.List[ConversionMatrix]:
    stacked_coefficients = np.vstack([nlc.lc.coefficients for nlc in named_lcs])
    conversion_matrices: typing.List[ConversionMatrix] = []
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
            ConversionMatrix(
                left.indices, moindex_left, right.indices, moindex_right, stacked_coefficients
            )
        )
    return conversion_matrices

