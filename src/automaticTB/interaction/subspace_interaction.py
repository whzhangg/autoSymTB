"""
This script is an experimental self-containing solver for the interactions, 
given the so called AOsubspace pair
"""
import typing, dataclasses
import numpy as np
from .interaction_pairs import AOPair, AOPairWithValue
from ..tools import LinearEquation
from .subspaces import InteractingAOSubspace

__all__ = [
    "CombinedAOSubspaceInteraction"
]

class CombinedAOSubspaceInteraction:
    def __init__(self, subspace_interactions: typing.List[InteractingAOSubspace]) -> None:
        self.all_AOpairs: typing.List[AOPair] = []
        for subspace in subspace_interactions:
            self.all_AOpairs += subspace.all_AOpairs
        
        self.considered_ao_pairs: typing.List[AOPair] = []
        for subspace in subspace_interactions:
            for aop in subspace.all_AOpairs:
                if aop not in self.considered_ao_pairs:
                    self.considered_ao_pairs.append(aop)

        num_homogeneous = sum(
            [len(sub.linear_equation.homogeneous_equation) for sub in subspace_interactions]
        )
        num_non_homogen = sum(
            [len(sub.linear_equation.non_homogeneous) for sub in subspace_interactions]
        )

        homogeneous_part = np.zeros(
            (num_homogeneous, len(self.considered_ao_pairs)),
            dtype = subspace_interactions[0].linear_equation.full_equation.dtype
        )
        non_homogeneous = np.zeros(
            (num_non_homogen, len(self.considered_ao_pairs)),
            dtype = subspace_interactions[0].linear_equation.full_equation.dtype
        )

        row_homo_start = 0
        row_nonh_start = 0
        for subspace in subspace_interactions:
            all_index = [self.considered_ao_pairs.index(aop) for aop in subspace.all_AOpairs]
            row_homo_end = row_homo_start + len(subspace.linear_equation.homogeneous_equation)
            
            if len(subspace.linear_equation.homogeneous_equation) > 0:
                homogeneous_part[row_homo_start:row_homo_end, all_index] \
                    = subspace.linear_equation.homogeneous_equation
            row_homo_start = row_homo_end
            
            row_nonh_end = row_nonh_start + len(subspace.linear_equation.non_homogeneous)
            if len(subspace.linear_equation.non_homogeneous) > 0:
                non_homogeneous[row_nonh_start:row_nonh_end, all_index] \
                    = subspace.linear_equation.non_homogeneous
            row_nonh_start = row_nonh_end

        #self._num_free_parameter = np.linalg.matrix_rank(non_homogeneous, tol = zero_tolerance)
        #homogeneous_rank = np.linalg.matrix_rank(homogeneous_part, tol = zero_tolerance)
        #if homogeneous_rank + self._num_free_parameter \
        #            != len(self.considered_ao_pairs):
        #    print(f"rank of homogeneous equation: {homogeneous_rank}")
        #    print(f"rank of non-homogeneous part: {self._num_free_parameter}")
        #    print(f"total number of AO in the equation: {len(self.considered_ao_pairs)}")
        self.homogeneous_equation = LinearEquation(homogeneous_part)

    @property
    def free_AOpairs(self) -> typing.List[AOPair]:
        return [
            self.considered_ao_pairs[i] for i in self.homogeneous_equation.free_variables_index
        ]
        
    def solve_interactions_to_InteractionPairs(
        self, values: typing.List[float]
    ) -> typing.List[AOPairWithValue]:
        values = np.array(values)

        all_solved_values = self.homogeneous_equation.solve_providing_values(values)
        assert len(all_solved_values) == len(self.considered_ao_pairs)
        aopairs_withvalue = []
        for pair_to_solve in self.all_AOpairs:
            where = self.considered_ao_pairs.index(pair_to_solve)
            aopairs_withvalue.append(
                AOPairWithValue(pair_to_solve.l_AO, pair_to_solve.r_AO, all_solved_values[where])
            )

        return aopairs_withvalue