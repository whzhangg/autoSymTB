import dataclasses, typing, collections
import numpy as np
from ..utilities import Pair, PairwithValue, tensor_dot
from ..structure import NearestNeighborCluster
from ..interaction import InteractionPairs
from ..structure.nncluster import ClusterSubSpace
from ..parameters import zero_tolerance
from ..SALCs.vectorspace import VectorSpace
from ..SALCs.linear_combination import LinearCombination
from ..atomic_orbitals import AO


class NamedLC(typing.NamedTuple):
    name: str
    lc: LinearCombination


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

        



def get_rep_LC_from_decomposed_vectorspace_as_dict(vc_dicts: typing.Dict[str, VectorSpace]) -> typing.List[NamedLC]:
    result = []
    for irrepname, vs in vc_dicts.items():
        lcs = vs.get_nonzero_linear_combinations()
        for ilc in range(vs.rank):
            result.append(NamedLC(irrepname, lcs[ilc].get_normalized()))
    return result


def divide_by_subspace_pairs(coefficient: np.ndarray, subspacepair: Pair) \
-> Pair: # list of (i, "s"), i is the index of equivalent atom in the sites
    # make a list of slices for each atom and each orbitals
    left: ClusterSubSpace = subspacepair.left
    right: ClusterSubSpace = subspacepair.right

    row_left = []
    row_right = []
    for row in coefficient:
        if np.linalg.norm(row[left.indices]) > zero_tolerance:
            # if this subspace is non-zero, then other coefficients would be necessary zero
            row_left.append(row)
        if np.linalg.norm(row[right.indices]) > zero_tolerance:
            row_right.append(row)
    return Pair(
        np.vstack(row_left), 
        np.vstack(row_right)
    )

