"""
This script is an experimental self-containing solver for the interactions, 
given the so called AOsubspace pair
"""
import typing, numpy as np, dataclasses

from automaticTB.tools import LinearEquation
from automaticTB.parameters import complex_coefficient_type, zero_tolerance
from automaticTB.solve.structure import CenteredCluster
from .ao_rotation_tools import AOpairRotater, find_rotation_between_two_clusters, get_translated_AO
from .interface import generate_aopair_from_cluster
from .interaction_pairs import AOPair, AOPairWithValue
from .interaction_base import InteractionBase, SimpleInteraction


class CombinedEquivalentInteractingAO(InteractionBase):
    """this class use symmetry to establish the relationship of all orbital interactions
    
    It provide the same interface as the interaction base objects. 
    However, one important difference is that the free pairs are given by a "solvable part" to obtain additional interactions due to the conjugate relationship
    """
    def __init__(self,
        solved_interactions: typing.List[InteractionBase],
        nonequivalent_clustersets: typing.List[typing.List[CenteredCluster]],
        possible_rotations: np.ndarray
    ) -> None:
        if len(solved_interactions) != len(nonequivalent_clustersets):
            print("CombinedEquivalentInteraction ",
                f"input number solution {len(solved_interactions)} is different from " ,
                f"the number of non-equivalent clusters {len(nonequivalent_clustersets)} !!")
            raise RuntimeError
        
        self._all_aopairs = []
        for equivalent_clusters in nonequivalent_clustersets:
            for cluster in equivalent_clusters:
                related_pairs = generate_aopair_from_cluster(cluster)
                self._all_aopairs += related_pairs

        self.num_aopairs = len(self._all_aopairs)        
        self._aopair_index: typing.Dict[AOPair, int] = {
            ao: iao for iao, ao in enumerate(self._all_aopairs)
        }

        # generate the known homogeneous matrix by stacking them
        self._homogeneous_relationship: typing.List[np.ndarray] = []
        for interaction in solved_interactions:
            _block = np.zeros(
                (len(interaction.homogeneous_matrix),self.num_aopairs), 
                dtype=complex_coefficient_type
            )
            for ip, pair in enumerate(interaction.all_AOpairs):
                new_index = self._aopair_index[pair]
                _block[:, new_index] = interaction.homogeneous_matrix[:, ip]
            self._homogeneous_relationship.append(np.array(_block))
        
        # generate the rotational equivalence inside the unitcell
        self._symmetry_operations = []
        self._symmetry_relationships: typing.List[np.ndarray] = []
        self._symmetry_reverse_relationships: typing.List[np.ndarray] = []
        rotator = AOpairRotater(self._all_aopairs)

        for aointeractions, eq_clusters in zip(solved_interactions, nonequivalent_clustersets):
            # each of the cluster in the unit cell
            operations = []
            for cluster in eq_clusters:
                rotation = find_rotation_between_two_clusters(
                    eq_clusters[0], cluster, possible_rotations
                )
                frac_translation = cluster.center_site.absolute_position \
                                - eq_clusters[0].center_site.absolute_position
                operations.append((rotation, frac_translation))
                
                for aopair in aointeractions.all_AOpairs:
                    sym_relation, sym_rev_relation = rotator.rotate(
                        aopair, rotation, frac_translation, print_debug=False
                    )
                    if not np.all(np.isclose(sym_relation, 0.0, atol=zero_tolerance)):
                        self._symmetry_relationships.append(sym_relation)
                    if not np.all(np.isclose(sym_rev_relation, 0.0, atol=zero_tolerance)):
                        self._symmetry_reverse_relationships.append(sym_rev_relation)
            
            self._symmetry_operations.append(operations)
                
        self._homogeneous_equation = LinearEquation.from_equation(np.vstack(
                self._homogeneous_relationship + \
                self._symmetry_relationships
            ))

        equation_with_additional_relationship = LinearEquation.from_equation(np.vstack(
            self._homogeneous_relationship + \
            self._symmetry_relationships + \
            self._symmetry_reverse_relationships
        ))

        free_pair = [self._all_aopairs[i] \
                    for i in equation_with_additional_relationship.free_variable_indices]
                    
        self.solvable_interaction = self.find_solvable_part(free_pair)
    
    def find_solvable_part(self, provided_aopairs: typing.List[AOPair]) -> InteractionBase:
        """find the part of equation that is solvable provided free pairs"""
        true_matrix = self._homogeneous_equation.row_echelon_form

        new_rows = np.zeros(
            (len(provided_aopairs), len(self.all_AOpairs)), dtype=complex_coefficient_type)

        for i, pair in enumerate(provided_aopairs):
            which_pair = self._aopair_index[pair]
            new_rows[i, which_pair] = 1.0

        tmp = LinearEquation.from_equation(np.vstack([true_matrix, new_rows]))
        index = np.array(tmp.free_variable_indices)
        #print(index)
        if len(index) + len(provided_aopairs) != \
            len(self._homogeneous_equation.free_variable_indices):
            print("============")
            print(f"number of provided aopairs {len(provided_aopairs)} + {len(index)} additional free index != total number of values ({len(self.free_AOpairs)}) needed")
            raise RuntimeError
        assert len(index) + len(provided_aopairs) == \
            len(self._homogeneous_equation.free_variable_indices)

        #print(true_matrix.shape)
        # if a row has any of the index non zero, then all the non-zero ones should be removed
        related_indices = set()
        for row in true_matrix:
            if np.all(np.isclose(row[index], 0.0, atol = zero_tolerance)):
                continue
            #print(np.argwhere(np.invert(np.isclose(row, 0.0, atol=zero_tolerance))).flatten())
            related_indices |= set(
                np.argwhere(np.invert(np.isclose(row, 0.0, atol=zero_tolerance))).flatten()
            )
        
        #print(related_indices)
        new_coloms = []
        solvable_aopairs = []
        for icol, pair in enumerate(self.all_AOpairs):
            if icol not in related_indices:
                new_coloms.append(true_matrix[:, icol])
                solvable_aopairs.append(pair)

        solvable_matrix = LinearEquation.from_equation(np.vstack(new_coloms).T)
        #print(solvable_matrix.row_echelon_form.shape)
        if len(solvable_matrix.free_variable_indices) != len(provided_aopairs):
            print("created solvable matrix is problematic, with number of free variable ",
                len(solvable_matrix), f" but with {len(provided_aopairs)} input values")
            raise RuntimeError
        return SimpleInteraction(solvable_aopairs, solvable_matrix.row_echelon_form)    
        
        
    @property
    def homogeneous_equation(self) -> typing.Optional[LinearEquation]:
        return self._homogeneous_equation

    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_aopairs

    @property
    def free_AOpairs(self) -> typing.List[AOPair]:
        return self.solvable_interaction.free_AOpairs


    def solve_interactions_to_InteractionPairs(
        self, values: typing.List[float]
    ) -> typing.List[AOPairWithValue]:

        solved_aopairs = self.solvable_interaction.solve_interactions_to_InteractionPairs(values)
        solved_values_dict = {solved_pair: solved_pair.value for solved_pair in solved_aopairs}

        values = []
        for freepair in self.free_AOpairs:
            reverse_free_pair = get_translated_AO(AOPair(freepair.r_AO, freepair.l_AO))
            if freepair in solved_values_dict:
                values.append(solved_values_dict[freepair])
            elif reverse_free_pair in solved_values_dict:
                values.append(np.conjugate(solved_values_dict[reverse_free_pair]))
            else:
                raise RuntimeError("interaction pair not found")

        super().solve_interactions_to_InteractionPairs(values)


    def print_log(self) -> None:
        print('## Combined Equivalent Interaction of the whole system', end = ' ')
        print(f"  there are {len(self._symmetry_operations)} non-equivalent sites")
        print(f"  all position can be generated by the following operations:")
        for ineq, sym_ops in enumerate(self._symmetry_operations):
            print(f"    equivalent set {ineq + 1:>2d} -------")
            for iop, (rot, trans) in enumerate(sym_ops):
                print(
                    f"    {iop+1} rot: ", 
                    ("{:>6.2f}"*9).format(*rot.flatten()),
                    " tran = ", 
                    ("{:>6.2f}"*3).format(*trans),
                )
        print("")
        print("  Num. of orbital relationship due to point group symmetry: ")
        for ineq, homo in enumerate(self._homogeneous_relationship):
            print(f"    {ineq+1:>2d} non-equivalent position : {len(homo)}")
        
        print(f"  Num. of rot./trans. relationship: {len(self._symmetry_relationships)}")
        print(f"  Num. of reverse relationship: {len(self._symmetry_reverse_relationships)}")
        print("")
        print("Combined Equivalent Interaction Done !!")
