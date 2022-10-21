from .equation_system import InteractionEquation
from ..atomic_orbitals import AO
from .interaction_pairs import AOPair, InteractionPairs
import dataclasses, typing
import numpy as np
from ..tools import find_free_variable_indices_by_row_echelon, tensor_dot

__all__ = ["CombinedInteractionEquation"]

class CombinedInteractionEquation:
    def __init__(self, cluster_equations: typing.List[InteractionEquation]) -> None:
        # the interaction ao pairs
        self.interaction_ao_pairs: typing.List[AOPair] = []
        for equation in cluster_equations:
            self.interaction_ao_pairs += [
                aop for aop in equation.ao_pairs if aop not in self.interaction_ao_pairs
            ]

        # interaction AO homogeneous equations
        nrow = sum([len(equation.homogeneous_equation) for equation in cluster_equations])
        self.homogeneous_equation = np.zeros(
            (nrow, len(self.interaction_ao_pairs)), 
            dtype=cluster_equations[0].homogeneous_equation.dtype
        )
        row_start = 0
        for equation in cluster_equations:
            all_index = [self.interaction_ao_pairs.index(aop) for aop in equation.ao_pairs]
            row_end = row_start + len(equation.homogeneous_equation)
            self.homogeneous_equation[row_start:row_end, all_index] \
                = equation.homogeneous_equation
            row_start = row_end

        self.free_interaction_variable_indexs = \
            find_free_variable_indices_by_row_echelon(self.homogeneous_equation)
        
        # all the centered atomic orbitals on the same center
        self._self_AOpairs: typing.List[AOPair] = []
        all_centered_aos: typing.List[AO] = []
        for equation in cluster_equations:
            all_centered_aos += equation._center_aos
            self._self_AOpairs += tensor_dot(equation._center_aos, equation._center_aos)
        
        znl_list = list(set([(ao.atomic_number, ao.n, ao.l) for ao in all_centered_aos]))
        self._centered_free_AOpairs: typing.List[AOPair] = []
        for (z, n, l) in znl_list:
            for ao in all_centered_aos:
                if ao.atomic_number == z and ao.n == n and ao.l == l:
                    self._centered_free_AOpairs.append(AOPair(ao, ao))
                    break
        
    @property
    def free_AOpairs(self) -> typing.List[AOPair]:
        return self._centered_free_AOpairs + \
            [ self.interaction_ao_pairs[i] for i in self.free_interaction_variable_indexs ]
    
    def solve_interactions_to_InteractionPairs(
        self, values: typing.List[float]
    ) -> InteractionPairs:
        assert len(values) == len(self.free_AOpairs)
