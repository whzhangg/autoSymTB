import typing
import numpy as np
from ..atomic_orbitals import AO
from ..tools import LinearEquation
from .interaction_equation import InteractionEquation
from ._abstract_equation import AOEquationBase
from .interaction_pairs import AOPair, AOPairWithValue


__all__ = ["CombinedInteractionEquation"]


class CombinedInteractionEquation(AOEquationBase):
    """
    this class provide functionality that combine atomic orbital interaction on 
    different cluster and combine them into a single solver
    """
    def __init__(self, cluster_equations: typing.List[InteractionEquation]) -> None:
        # the interaction ao pairs
        self.interaction_ao_pairs: typing.List[AOPair] = []
        for equation in cluster_equations:
            self.interaction_ao_pairs += [
                aop for aop in equation.ao_pairs if aop not in self.interaction_ao_pairs
            ]

        # interaction AO homogeneous equations
        nrow = sum([len(equation.homogeneous_equation) for equation in cluster_equations])
        homogeneous_equation = np.zeros(
            (nrow, len(self.interaction_ao_pairs)), 
            dtype=cluster_equations[0].homogeneous_equation.dtype
        )
        row_start = 0
        for equation in cluster_equations:
            all_index = [self.interaction_ao_pairs.index(aop) for aop in equation.ao_pairs]
            row_end = row_start + len(equation.homogeneous_equation)
            homogeneous_equation[row_start:row_end, all_index] \
                = equation.homogeneous_equation
            row_start = row_end

        self.linear_equation = LinearEquation(homogeneous_equation)
        
        # all the centered atomic orbitals on the same center
        all_centered_aos: typing.List[AO] = []
        self._all_centered_AOpairs: typing.List[AOPair] = []
        for equation in cluster_equations:
            all_centered_aos += equation._center_aos
            for cao in equation._center_aos:
                self._all_centered_AOpairs.append(AOPair(cao, cao))
        znl_list = list(set([(ao.atomic_number, ao.n, ao.l) for ao in all_centered_aos]))

        self._centered_free_AOpairs: typing.List[AOPair] = []
        for (z, n, l) in znl_list:
            for ao in all_centered_aos:
                if ao.atomic_number == z and ao.n == n and ao.l == l:
                    self._centered_free_AOpairs.append(AOPair(ao, ao))
                    break

        self._AO_gathered: typing.List[AOPair] = []
        for equation in cluster_equations:
            self._AO_gathered += equation.all_AOpairs

    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_centered_AOpairs + self.interaction_ao_pairs

    @property
    def free_AOpairs(self) -> typing.List[AOPair]:
        return self._centered_free_AOpairs + \
            [ self.interaction_ao_pairs[i] for i in self.linear_equation.free_variables_index ]
    
    def solve_interactions_to_InteractionPairs(
        self, values: typing.List[float]
    ) -> typing.List[AOPairWithValue]:
        values = np.array(values)
        orbital_energy = values[0: len(self._centered_free_AOpairs)]
        interactions = values[len(self._centered_free_AOpairs): ]

        all_orbital_energies = []
        for orbital_pair in self._all_centered_AOpairs:
            for oe, free_centered_pair in zip(orbital_energy, self._centered_free_AOpairs):
                if orbital_pair.l_AO.atomic_number == free_centered_pair.l_AO.atomic_number and \
                   orbital_pair.l_AO.n == free_centered_pair.l_AO.n and \
                   orbital_pair.l_AO.l == free_centered_pair.l_AO.l:
                    all_orbital_energies.append(oe)
                    break

        assert len(interactions) == len(self.linear_equation.free_variables_index)
        all_interaction = list(self.linear_equation.solve_providing_values(interactions))
        assert len(all_interaction) == len(self.interaction_ao_pairs)

        all_solved_pairs = self._all_centered_AOpairs + self.interaction_ao_pairs
        all_solved_values = all_orbital_energies + all_interaction
        aopairs_withvalue = []
        for pair_to_solve in self._AO_gathered:
            where = all_solved_pairs.index(pair_to_solve)
            aopairs_withvalue.append(
                AOPairWithValue(pair_to_solve.l_AO, pair_to_solve.r_AO, all_solved_values[where])
            )

        return aopairs_withvalue