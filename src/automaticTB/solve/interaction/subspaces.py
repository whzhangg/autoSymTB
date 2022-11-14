import typing, dataclasses, re, numpy as np
from .interaction_pairs import AO, AOPair, AOPairWithValue
from ..SALCs import NamedLC
from ...tools import Pair, LinearEquation, tensor_dot, format_lines_into_two_columns
from ...parameters import zero_tolerance
from ..SALCs import IrrepSymbol

__all__ = ["AOSubspace", "InteractingAOSubspace"]


@dataclasses.dataclass
class AOSubspace:
    aos: typing.List[AO]
    namedlcs: typing.List[NamedLC] # only the components belonging to the AOs

    def __repr__(self) -> str:
        result = ["subspace consists of:"]
        for ao in self.aos:
            result.append(f" {str(ao)}")
        result.append("With linear combinations")
        for nlc in self.namedlcs:
            result.append(f" {str(nlc.name)}")
        return "\n".join(result)


class FullLinearEquation:
    """
    This class contain the matrix equation of the given set of interaction
    """
    _forbidden = 0

    def __init__(self, tps: np.ndarray, labels: typing.List[int]) -> None:
        assert len(tps) > 0
        non_homogeneous  = []
        homogeneous_part = [tps[i] for i, type in enumerate(labels) if type == self._forbidden]
        all_types = set(labels) - set([self._forbidden])
        for thetype in all_types:
            indices = [i for i, type in enumerate(labels) if type == thetype]
            if len(indices) > 1:
                for i in indices[1:]:
                    homogeneous_part.append(tps[i] - tps[indices[0]])
            non_homogeneous.append(tps[indices[0]])
        
        assert len(homogeneous_part) + len(non_homogeneous) == tps.shape[1]
        if len(homogeneous_part) == 0:
            self.full_equation = np.array(non_homogeneous)
            self._homogeneous_indices = np.array([])
            self._non_homogeneous_indices = np.arange(len(self.full_equation))
        elif len(non_homogeneous) == 0:
            self.full_equation = np.array(homogeneous_part)
            self._homogeneous_indices = np.arange(len(self.full_equation))
            self._non_homogeneous_indices = np.array([])
        else:
            self.full_equation = np.vstack([homogeneous_part, non_homogeneous])
            self._homogeneous_indices = np.arange(len(homogeneous_part))
            self._non_homogeneous_indices = np.arange(len(non_homogeneous)) + len(homogeneous_part)

        self._num_free_parameter = len(self._non_homogeneous_indices)

        self._homogeneous_solver = None
        if len(self._homogeneous_indices) > 0:
            self._homogeneous_solver = LinearEquation(
                self.full_equation[self._homogeneous_indices, :]
            )

    @property
    def homogeneous_equation(self) -> np.ndarray:
        if len(self._homogeneous_indices) > 0:
            return self.full_equation[self._homogeneous_indices, :]
        else:
            return np.array([])
    
    @property
    def non_homogeneous(self) -> np.ndarray:
        if len(self._non_homogeneous_indices) > 0:
            return self.full_equation[self._non_homogeneous_indices, :]
        else:
            return np.array([])

    @property
    def free_variables_index(self) -> typing.List[int]:
        if len(self._homogeneous_indices) == 0:
            larger_then_zero = \
                np.argwhere(np.linalg.norm(self.non_homogeneous, axis=1) > zero_tolerance)
            assert len(larger_then_zero) == len(self._non_homogeneous_indices)
            return larger_then_zero[0]

        else:
            return self._homogeneous_solver.free_variables_index

    def solve_providing_values(
        self, free_interaction_values: typing.List[float]
    ) -> np.ndarray:
        assert len(free_interaction_values) == self._num_free_parameter
        if len(self._homogeneous_indices) == 0:
            return free_interaction_values
        else:
            return self._homogeneous_solver.free_variables_index   
        
    @classmethod
    def from_representations_tp_list(cls, 
        reps_tp_list: typing.List[typing.Tuple[IrrepSymbol, IrrepSymbol, np.ndarray]]
    ) -> "FullLinearEquation":
        references = []
        tps = [tp for _, _, tp in reps_tp_list]
        memory = {}
        symbols = 1
        for rep1, rep2, _ in reps_tp_list:
            if rep1.symmetry_symbol != rep2.symmetry_symbol:
                references.append(cls._forbidden)
            else:
                main1 = f"{rep1.main_irrep}^{rep1.main_index}"
                main2 = f"{rep2.main_irrep}^{rep2.main_index}"
                pair = " ".join([main1, main2])
                if pair not in memory:
                    memory[pair] = symbols
                    references.append(symbols)
                    symbols += 1
                else:
                    references.append(memory[pair])
        
        return cls(np.array(tps), np.array(references))


class InteractingAOSubspace:
    """
    just a block form of Hamiltonian between two subspace
    """
    def __init__(self, l_subspace: AOSubspace, r_subspace: AOSubspace) -> None:
        self.l_subspace = l_subspace
        self.r_subspace = r_subspace
        self.ao_pair = [ 
            AOPair.from_pair(p) for p in  tensor_dot(l_subspace.aos, r_subspace.aos) 
        ]

        reps_tp_list = []
        for l_nlc in l_subspace.namedlcs:
            for r_nlc in r_subspace.namedlcs:
                l_rep = l_nlc.name
                r_rep = r_nlc.name
                tp = np.tensordot(
                    l_nlc.lc.coefficients, 
                    r_nlc.lc.coefficients,
                    axes=0
                ).flatten()
                reps_tp_list.append(
                    (l_rep, r_rep, tp)
                )

        self.linear_equation = FullLinearEquation.from_representations_tp_list(
                                    reps_tp_list
                                )

    @property
    def free_AOpairs(self) -> typing.List[AOPair]:
        return [self.ao_pair[i] for i in self.linear_equation.free_variables_index]


    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self.ao_pair

        
    def solve_interactions(self, values: typing.List[float]) -> typing.List[AOPairWithValue]:
        if len(values) != len(self.free_AOpairs):
            print(f"the number of input parameter ({len(values)}) is different") 
            print(f"   from the number of indices ({len(self.free_AOpairs)})")

        aopairs_withvalue = []
        all_values = self._linear_equation.solve_providing_values(values)

        for aopair, v in zip(self.all_AOpairs, all_values):
            aopairs_withvalue.append(
                AOPairWithValue(aopair.l_AO, aopair.r_AO, v)
            )

        return aopairs_withvalue

    def __str__(self) -> str:
        l_str = {0:"s", 1:"p", 2:"d", 3:"f"}
        l_ele = self.ao_pair[0].l_AO.chemical_symbol
        l_orb = f"{self.ao_pair[0].l_AO.n:>2d}{l_str[self.ao_pair[0].l_AO.l]}"
        r_ele = self.ao_pair[0].r_AO.chemical_symbol
        r_orb = f"{self.ao_pair[0].r_AO.n:>2d}{l_str[self.ao_pair[0].r_AO.l]}"
        vector = self.ao_pair[0].r_AO.absolute_position - self.ao_pair[0].l_AO.absolute_position
        distance = np.linalg.norm(vector)
        part1 = ["-------------------------------------------------------"]
        part1.append(f"Block: ({l_ele:>2s}{l_orb}) -- r ={distance:>6.3}A --> ({r_ele:<2s}{r_orb})")
        l_lines = ["left"]
        for nlc in self.l_subspace.namedlcs:
            l_lines.append(str(nlc.name))
        r_lines = ["right"]
        for nlc in self.r_subspace.namedlcs:
            r_lines.append(str(nlc.name))
        part1.append(format_lines_into_two_columns(l_lines, r_lines))
        part1.append("-------------------------------------------------------")
        return "\n".join(part1)

