"""
This script is an experimental self-containing solver for the interactions, 
given the so called AOsubspace pair
"""
import typing, numpy as np

from automaticTB.tools import LinearEquation
from automaticTB.parameters import complex_coefficient_type, zero_tolerance
from automaticTB.solve.structure import CenteredCluster
from .ao_rotation_tools import AOpairRotater, find_rotation_between_two_clusters
from .interface import generate_aopair_from_cluster
from .interaction_pairs import AOPair
from .interaction_base import InteractionBase


class CombinedEquivalentInteractingAO(InteractionBase):
    """this class use symmetry to establish the relationship of all orbital interactions"""
    def __init__(self,
        solved_interactions: typing.List[InteractionBase],
        nonequivalent_clustersets: typing.List[typing.List[CenteredCluster]],
        possible_rotations: np.ndarray,
        reverse_equivalence: bool
    ) -> None:
        self._use_reverse_equivalence = reverse_equivalence
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
        aopair_index: typing.Dict[AOPair, int] = {
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
                new_index = aopair_index[pair]
                _block[:, new_index] = interaction.homogeneous_matrix[:, ip]
            self._homogeneous_relationship.append(np.array(_block))
        
        # generate the rotational equivalence inside the unitcell
        symmetry_operations = []
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
                
        # 
        if reverse_equivalence:
            self._homogeneous_equation = LinearEquation(np.vstack(
                self._homogeneous_relationship + \
                self._symmetry_relationships + \
                self._symmetry_reverse_relationships
            ))
        else:
            self._homogeneous_equation = LinearEquation(np.vstack(
                self._homogeneous_relationship + \
                self._symmetry_relationships
            ))


    @property
    def homogeneous_equation(self) -> typing.Optional[LinearEquation]:
        return self._homogeneous_equation


    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_aopairs


    def print_log(self) -> None:
        print('## Combined Equivalent Interaction of the whole system', end = ' ')
        if self._use_reverse_equivalence:
            print('(assume relationship Eij = Eji)')
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
        if self._use_reverse_equivalence:
            print(f"  Num. of reverse relationship: {len(self._symmetry_reverse_relationships)}")
        print("")
        print("Combined Equivalent Interaction Done !!")
