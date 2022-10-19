import typing, dataclasses
from ..atomic_orbitals import AO    
import numpy as np
from ..tools import find_free_variable_indices, tensor_dot, Pair
from ..structure import NearestNeighborCluster, ClusterSubSpace
from ..parameters import zero_tolerance
from ..SALCs import NamedLC, IrrepSymbol
from .interaction_pairs import InteractionPairs
from .MOInteraction import HomogeneousEquationFinder


__all__ = ["InteractionEquation"]


@dataclasses.dataclass
class InteractionEquation:
    def __init__(self,
        ao_pairs: typing.List[AO],
        homogeneous_equation: np.ndarray
    ) -> None:
        self.ao_pairs = ao_pairs
        self.homogeneous_equation = homogeneous_equation
        self.free_variables_indices: typing.List[int] \
            = list(find_free_variable_indices(homogeneous_equation))
        
        assert len(self.ao_pairs) == \
            len(self.homogeneous_equation) + len(self.free_variables_indices)

    @classmethod
    def from_nncluster_namedLC(
        cls, nncluster: NearestNeighborCluster, named_lcs: typing.List[NamedLC]
    ) -> "InteractionEquation":
        aos = nncluster.AOlist
        ao_indices = list(range(len(aos)))
        mo_indices = list(range(len(named_lcs)))
        print(ao_indices)
        print(mo_indices)
        assert len(ao_indices) == len(mo_indices)

        left_ao_indices = set()
        right_ao_indices = set()
        for subspace_pair in nncluster.interaction_subspace_pairs:
            left: ClusterSubSpace = subspace_pair.left
            right: ClusterSubSpace = subspace_pair.right
            left_ao_indices |= set(left.indices)
            right_ao_indices |= set(right.indices)
        left_ao_indices = sorted(list(left_ao_indices))
        right_ao_indices = sorted(list(right_ao_indices))

        homogeneous = []
        for subspace_pair in nncluster.interaction_subspace_pairs:
            left: ClusterSubSpace = subspace_pair.left
            right: ClusterSubSpace = subspace_pair.right

            moindex_left = []
            moindex_right = []
            for i, namedlc in enumerate(named_lcs):
                if np.linalg.norm(namedlc.lc.coefficients[left.indices]) > zero_tolerance:
                    # if this subspace is non-zero, other coefficients would be necessary zero
                    moindex_left.append(i)
                if np.linalg.norm(namedlc.lc.coefficients[right.indices]) > zero_tolerance:
                    moindex_right.append(i)

            finder = HomogeneousEquationFinder()
            for lmo in moindex_left:
                for rmo in moindex_right:
                    rep1 = named_lcs[lmo].name
                    rep2 = named_lcs[rmo].name
                    tp = np.tensordot(
                        named_lcs[lmo].lc.coefficients[left_ao_indices],
                        named_lcs[rmo].lc.coefficients[right_ao_indices],
                        axes=0
                    ).flatten()
                    found = finder.get_equation(rep1, rep2, tp)
                    if not (found is None):
                        homogeneous.append(found)
        
        homogeneous = np.array(homogeneous)

        aopairs = tensor_dot(
            [aos[i] for i in left_ao_indices],
            [aos[i] for i in right_ao_indices]
        )
        return cls(aopairs, np.array(homogeneous))

    @property
    def num_AOpairs(self) -> int:
        return len(self.ao_pairs)

    @property
    def num_free_variables(self) -> int:
        return len(self.free_variables_indices)

    @property
    def free_AOpairs(self) -> typing.List[Pair]:
        return [ self.ao_pairs[i] for i in self.free_variables_indices ]


    def solve_interactions_to_InteractionPairs(
        self, values: typing.List[float]
    ) -> InteractionPairs:
        # we assume that the sequence is correct.
        assert len(values) == len(self.free_variables_indices)

        additional_A = np.zeros(
            (self.num_free_variables, self.num_AOpairs), 
            dtype=self.homogeneous_equation.dtype
        )

        for i, which in enumerate(self.free_variables_indices):
            additional_A[i, which] = 1.0
        additional_b = np.array(values)
        original_b = np.zeros(
            self.num_AOpairs - self.num_free_variables, dtype=additional_b.dtype
        )

        b = np.hstack([original_b, additional_b])
        A = np.vstack([self.homogeneous_equation, additional_A])
        assert len(A) == len(b)
        assert A.shape[0] == A.shape[1]
        inv_A = np.linalg.inv(A)

        return InteractionPairs(self.ao_pairs, np.dot(inv_A, b))
        
