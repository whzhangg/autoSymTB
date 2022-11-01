import numpy as np
import typing
from ..tools import find_free_variable_indices_by_row_echelon, tensor_dot, Pair
from ..tools import LinearEquation
from ..structure import NearestNeighborCluster, ClusterSubSpace
from ..parameters import zero_tolerance
from ..SALCs import NamedLC, IrrepSymbol
from .interaction_pairs import AO, AOPair, AOPairWithValue
from ._abstract_equation import AOEquationBase


__all__ = ["InteractionEquation"]


class HomogeneousEquationFinder:
    """
    a class that help to organize the homogenous equation from the interaction rule
    """
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


class InteractionEquation(AOEquationBase):
    """
    This class keep track of the interactions, given a nncluster
    It treat AO interactions separately:
    1. one set is the energy of the isolated atomic energy, 
    2. the other is the interaction energy
    The main interface is just:
    - from_nncluster_namedLC()
    - free_AOpairs
    - solve_interactions_to_InteractionPairs
    Other internal variables are not going to be accessed in the usual case
    """
    def __init__(self,
        center_aos: typing.List[AO],
        interaction_ao_pairs: typing.List[Pair],
        homogeneous_equation: np.ndarray
    ) -> None:
        self.center_aos = center_aos
        self._centered_free_AOpairs: typing.List[AOPair] = []
        for (n,l) in list(set([ (ao.n, ao.l) for ao in self.center_aos ])):
            for ao in self.center_aos:
                if ao.n == n and ao.l == l:
                    self._centered_free_AOpairs.append(AOPair(ao, ao))
                    break
        
        self.ao_pairs = [AOPair.from_pair(ao_pair) for ao_pair in interaction_ao_pairs] 
        self.homogeneous_equation = homogeneous_equation

        self.free_variables_indices: typing.List[int] \
            = list(find_free_variable_indices_by_row_echelon(homogeneous_equation))
        
        assert len(self.ao_pairs) == \
            len(self.homogeneous_equation) + len(self.free_variables_indices)

    @classmethod
    def from_nncluster_namedLC(
        cls, nncluster: NearestNeighborCluster, named_lcs: typing.List[NamedLC]
    ) -> "InteractionEquation":
        aos = nncluster.AOlist
        ao_indices = list(range(len(aos)))
        mo_indices = list(range(len(named_lcs)))
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
        return cls(nncluster.center_AOs, aopairs, np.array(homogeneous))


    @property
    def free_AOpairs(self) -> typing.List[AOPair]:
        """
        return the free AO pairs, it's a combination of free atomic orbital energy 
        as well as the free interactions. Atomic orbital energys are treated separated from the 
        interactions
        """
        return self._centered_free_AOpairs + \
            [ self.ao_pairs[i] for i in self.free_variables_indices ]

    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        centered_ao_pairs = []
        for l_ao in self.center_aos:
            centered_ao_pairs.append(AOPair(l_ao, l_ao))
        return centered_ao_pairs + self.ao_pairs


    def solve_interactions_to_InteractionPairs(
        self, values: typing.List[float]
    ) -> typing.List[AOPairWithValue]:
        # we assume that the sequence is correct.
        assert len(values) == len(self.free_AOpairs)

        all_ao_pairs: typing.List[AOPair] = []
        all_values = []
        nl_value = {}
        for i, c_free_aop in enumerate(self._centered_free_AOpairs):
            n, l = c_free_aop.l_AO.n, c_free_aop.l_AO.l
            nl_value[(n, l)] = values[i]

        for l_ao in self.center_aos:
            all_ao_pairs.append(AOPair(l_ao, l_ao))
            all_values.append(nl_value[(l_ao.n, l_ao.l)])
            #for r_ao in self.center_aos:
            #    all_ao_pairs.append(AOPair(l_ao, r_ao))
            #    if l_ao.n == r_ao.n and l_ao.l == r_ao.l and l_ao.m == r_ao.m:
            #        all_values.append(nl_value[(l_ao.n, l_ao.l)])
            #    else:
            #        all_values.append(0.0)

        additional_A = np.zeros(
            (len(self.free_variables_indices), len(self.ao_pairs)), 
            dtype=self.homogeneous_equation.dtype
        )

        for i, which in enumerate(self.free_variables_indices):
            additional_A[i, which] = 1.0
        additional_b = np.array(values[len(self._centered_free_AOpairs):])
        original_b = np.zeros(
            len(self.ao_pairs) - len(self.free_variables_indices), dtype=additional_b.dtype
        )

        b = np.hstack([original_b, additional_b])
        A = np.vstack([self.homogeneous_equation, additional_A])
        assert len(A) == len(b)
        assert A.shape[0] == A.shape[1]
        inv_A = np.linalg.inv(A)

        all_ao_pairs += self.ao_pairs
        all_values += list(np.dot(inv_A, b))

        aopairs_withvalue = []
        for aopair, v in zip(all_ao_pairs, all_values):
            aopairs_withvalue.append(
                AOPairWithValue(aopair.l_AO, aopair.r_AO, v)
            )
        return aopairs_withvalue
        
