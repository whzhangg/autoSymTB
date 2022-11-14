"""
This script is an experimental self-containing solver for the interactions, 
given the so called AOsubspace pair
"""
import typing, dataclasses
import numpy as np
from .interaction_pairs import AOPair, AOPairWithValue
from ...tools import LinearEquation
from .subspaces import InteractingAOSubspace
from ...parameters import zero_tolerance

__all__ = [
    "CombinedAOSubspaceInteraction"
]


def get_additiona_equation(
    all_AOpairs: typing.List[AOPair],
    considered_ao_pairs: typing.List[AOPair],
    homogeneous_part: np.ndarray
) -> typing.Tuple[typing.List[AOPair], np.ndarray]:
    """
    This function organize an larger equation system that will yield all the values of AO pairs
    [[a11, a12, a13],  [x1,   [b1,
     [a21, a22, a23], * x2, =  b2,
     [a31, a32, a33]]   x3]    b3]

    Now we introduce a parameter x4 that is equivalent to x2, for example, we can rewrite the 
    relationship to:
    [[a11, a12, a13, 0],  [x1,   [b1,
     [a21, a22, a23, 0], * x2, =  b2,
     [a31, a32, a33, 0],   x3,    b3,
     [  0,   1,   0,-1]]   x4]     0]
    """
    num_all_pairs = len(all_AOpairs)
    num_considered = len(considered_ao_pairs)

    nrow, ncol = homogeneous_part.shape

    additional_row = num_all_pairs - num_considered
    new_matrix = np.zeros(
        (nrow + additional_row, ncol + additional_row), dtype=homogeneous_part.dtype
    )
    new_matrix[0:nrow, 0:ncol] = homogeneous_part

    additional_pair = list(set(all_AOpairs) - set(considered_ao_pairs))
    additional_indices = [ considered_ao_pairs.index(ap) for ap in additional_pair ]
    for i, equivalent_index in enumerate(additional_indices):
        new_matrix[nrow + i, equivalent_index] = 1.0
        new_matrix[nrow + i, ncol + i] = -1.0

    return considered_ao_pairs + additional_pair, new_matrix


class CombinedAOSubspaceInteraction:
    def __init__(self, subspace_interactions: typing.List[InteractingAOSubspace]) -> None:
        # gather all the AO pairs whose value we want
        self.all_AOpairs: typing.List[AOPair] = []
        for subspace in subspace_interactions:
            self.all_AOpairs += subspace.all_AOpairs
        
        # these are the AO pairs that enter the linear equation
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

        self.homogeneous_part = np.zeros(
            (num_homogeneous, len(self.considered_ao_pairs)),
            dtype = subspace_interactions[0].linear_equation.full_equation.dtype
        )
        self.non_homogeneous = np.zeros(
            (num_non_homogen, len(self.considered_ao_pairs)),
            dtype = subspace_interactions[0].linear_equation.full_equation.dtype
        )

        row_homo_start = 0
        row_nonh_start = 0
        for subspace in subspace_interactions:
            all_index = [self.considered_ao_pairs.index(aop) for aop in subspace.all_AOpairs]
            #print("* indices: " + str(all_index))
            row_homo_end = row_homo_start + len(subspace.linear_equation.homogeneous_equation)
            
            if len(subspace.linear_equation.homogeneous_equation) > 0:
                for i_org, i_new in enumerate(all_index):
                    self.homogeneous_part[row_homo_start:row_homo_end, i_new] \
                        += subspace.linear_equation.homogeneous_equation[:, i_org]
            row_homo_start = row_homo_end
            
            row_nonh_end = row_nonh_start + len(subspace.linear_equation.non_homogeneous)
            if len(subspace.linear_equation.non_homogeneous) > 0:
                for i_org, i_new in enumerate(all_index):
                    self.non_homogeneous[row_nonh_start:row_nonh_end, i_new] \
                        += subspace.linear_equation.non_homogeneous[:, i_org]
            row_nonh_start = row_nonh_end

        #self._num_free_parameter = np.linalg.matrix_rank(non_homogeneous, tol = zero_tolerance)
        #homogeneous_rank = np.linalg.matrix_rank(self.homogeneous_part, tol = zero_tolerance)
        #if homogeneous_rank + self._num_free_parameter \
        #            != len(self.considered_ao_pairs):
        #    print(f"rank of homogeneous equation: {homogeneous_rank}")
        #    print(f"rank of non-homogeneous part: {self._num_free_parameter}")
        #    print(f"total number of AO in the equation: {len(self.considered_ao_pairs)}")
        self.homogeneous_equation = LinearEquation(self.homogeneous_part)

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

    def print_debug_info(self,
        print_free: bool = False,
        print_considered: bool = False,
    ) -> None:
        print("*" * 60)
        print(f"total number of AO pairs needed: {len(self.all_AOpairs)}")
        print(f"      AO pairs enter the linear equation: {len(self.considered_ao_pairs)}")
        print("")
        homo_size = self.homogeneous_part.shape
        homo_rank = np.linalg.matrix_rank(self.homogeneous_part, tol = zero_tolerance)
        print(f"Homogeneous part size: ({homo_size[0]},{homo_size[1]})")
        print(f"                 rank: {homo_rank}")
        print("")
        nh_size = self.non_homogeneous.shape
        nh_rank = np.linalg.matrix_rank(self.non_homogeneous, tol = zero_tolerance)
        print(f"Non Homogen part size: ({nh_size[0]},{nh_size[1]})")
        print(f"                 rank: {nh_rank}")
        print("")
        if print_free:
            print(f"Number of free parameter: {len(self.free_AOpairs)}")
            for i, pair in enumerate(self.free_AOpairs):
                print(f"{i+1:>2d} " + str(pair))
        print("*" * 60)
        if print_considered:
            print(f"The considered pairs: ")
            for i, pair in enumerate(self.considered_ao_pairs):
                print(f"{i+1:>2d} " + str(pair))