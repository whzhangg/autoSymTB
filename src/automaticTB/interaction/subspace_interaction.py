"""
This script is an experimental self-containing solver for the interactions, 
given the so called AOsubspace pair
"""
import typing, dataclasses
import numpy as np
from .interaction_pairs import AO, AOPair, AOPairWithValue
from ..tools import tensor_dot
from ..tools import LinearEquation
from ..SALCs import IrrepSymbol
from ..parameters import zero_tolerance
from ..structure import NearestNeighborCluster
from ..SALCs import NamedLC, LinearCombination
from ..atomic_orbitals import Orbitals, OrbitalsList
from ..tools import Pair

__all__ = [
    "get_subspace_from_nncluster_linear_combination"
    "InteractingAOSubspace",
    "CombinedAOSubspaceInteraction"
]


_forbidden = 0


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


# a temperory interface
def get_subspace_from_nncluster_linear_combination(
    nncluster: NearestNeighborCluster,
    named_lcs: typing.List[NamedLC]
) -> typing.List[typing.Tuple[AOSubspace, AOSubspace]]:

    center_subspace = []
    other_subspaces = []

    for eq_type, atom_set in nncluster.equivalent_atoms_dict.items():
        for orb in nncluster.orbitalslist.orbital_list[eq_type].nl_list:
            ao_list = []
            sites = []
            orbitals = []
            indices = []
            for iatom in range(nncluster.num_sites):
                if iatom not in atom_set: continue
                sites.append(nncluster.crystalsites[iatom].site)
                orbital = Orbitals([orb])
                orbitals.append(orbital)
                for nlm in orbital.sh_list:
                    ao_list.append(
                        AO(
                            cluster_index=iatom, 
                            primitive_index=nncluster.crystalsites[iatom].index_pcell,
                            translation=nncluster.crystalsites[iatom].translation,
                            chemical_symbol=nncluster.crystalsites[iatom].site.chemical_symbol,
                            n = nlm.n,
                            l = nlm.l,
                            m = nlm.m
                        )
                    )
                slice = nncluster.orbitalslist.subspace_slice_dict[iatom][orb]
                indices += list(range(
                    slice.start, slice.stop
                ))
                
            lcs = []
            for namedlc in named_lcs:
                if np.linalg.norm(namedlc.lc.coefficients[indices]) > zero_tolerance:
                    lcs.append(NamedLC(
                        namedlc.name,
                        LinearCombination(
                            sites = sites, 
                            orbital_list = OrbitalsList(orbitals), 
                            coefficients = namedlc.lc.coefficients[indices]
                        )
                    ))
            assert len(ao_list) == len(lcs)

            subspace = AOSubspace(
                ao_list, lcs
            )

            if eq_type == nncluster.origin_index:
                center_subspace.append(subspace)
            
            other_subspaces.append(subspace)

    all_subspace_pairs = []
    for center in center_subspace:
        for other in other_subspaces:
            all_subspace_pairs.append(
                Pair(center, other)
            )

    return all_subspace_pairs


class FullLinearEquation:
    """
    This class contain the matrix equation of the given set of interaction
    """
    def __init__(self, tps: np.ndarray, labels: typing.List[int]) -> None:
        assert len(tps) > 0
        non_homogeneous  = []
        homogeneous_part = [tps[i] for i, type in enumerate(labels) if type == _forbidden]
        all_types = set(labels) - set([_forbidden])
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
                references.append(_forbidden)
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