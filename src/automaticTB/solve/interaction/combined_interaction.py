"""
This script is an experimental self-containing solver for the interactions, 
given the so called AOsubspace pair
"""
import typing
import numpy as np

from automaticTB.tools import LinearEquation
from automaticTB.parameters import tolerance_structure, complex_coefficient_type
from automaticTB.solve.structure import CenteredCluster
from automaticTB.solve.SALCs import VectorSpace
from automaticTB.solve.atomic_orbitals import OrbitalsList, Orbitals
from ._ao_rotation_tools import AOpairRotater
from .interface import _get_orbital_ln_from_string, _get_AO_from_CrystalSites_OrbitalList
from .interaction_pairs import AOPair
from .interaction_base import InteractionBase, InteractingAOSubspace


__all__ = ["BlockInteractions", "CombinedInteractingAO", "CombinedEquivalentInteractingAO"]


class BlockInteractions(InteractionBase):
    """r
    This class combineds the block diagonal form 
    """
    def __init__(self, interactions: typing.List[InteractionBase]) -> None:
        self._all_aopairs = []

        self._block_diagonal_size = []
        for interaction in interactions:
            self._all_aopairs += interaction.all_AOpairs
            homo_nrow = 0
            if interaction.homogeneous_equation is not None:
                homo_nrow = len(interaction.homogeneous_matrix)
            self._block_diagonal_size.append(
                (
                    len(interaction.all_AOpairs), 
                    homo_nrow
                )
            )

        nao = len(self._all_aopairs)
        nrow_homo = sum(n for _,n in self._block_diagonal_size)

        homo_matrix = np.zeros((nrow_homo, nao), dtype=complex_coefficient_type)

        nr1_start = 0 # homogeneous matrix
        col_start = 0
        for (ncol, nr1), interaction in zip(self._block_diagonal_size, interactions):
                
            nr1_end = nr1_start + nr1
            col_end = col_start + ncol

            if nr1 > 0:
                homo_matrix[nr1_start:nr1_end, col_start:col_end] \
                    = interaction.homogeneous_matrix

            col_start = col_end
            nr1_start = nr1_end
            
        self._homogeneous_equation = LinearEquation(homo_matrix)

    @property
    def homogeneous_equation(self) -> LinearEquation:
        return self._homogeneous_equation


    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_aopairs


    def print_log(self) -> None:
        print( '## Block Diagonal Interaction matrix')
        print( '  Homogeneous matrix is direct sum of:')
        tmp_out = []
        for ncol, nr1 in self._block_diagonal_size:
            tmp_out.append(f"({nr1:3>d} x {ncol:3>d})")
        print('  ' + " + ".join(tmp_out))
        print("")
        super().print_log()


class CombinedInteractingAO(InteractionBase):
    """
    This version only stores a single homogeneous equation instead of later finding 
    the values by matching interaction pairs. 

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

    @staticmethod
    def construct_homogeneous_part(
        subspace_interactions: typing.List[InteractingAOSubspace], 
        considered_ao_pairs: typing.List[AOPair],
        additional_pair: typing.List[AOPair]
    ) -> np.ndarray:
        """construct the homogeneous equation"""
        total_row = sum(
            [len(sub.linear_equation.homogeneous_equation) for sub in subspace_interactions]
        )

        homogeneous_part = np.zeros(
            (total_row, len(considered_ao_pairs)), 
            dtype = subspace_interactions[0].homogeneous_matrix.dtype
        )

        row_homo_start = 0
        for subspace in subspace_interactions:
            all_index = [considered_ao_pairs.index(aop) for aop in subspace.all_AOpairs]
            if subspace.homogeneous_equation is not None:
                row_homo_end = row_homo_start + len(subspace.homogeneous_matrix)
                for i_org, i_new in enumerate(all_index):
                    homogeneous_part[row_homo_start:row_homo_end, i_new] \
                        += subspace.linear_equation.homogeneous_equation[:, i_org]
            row_homo_start = row_homo_end

        nrow, ncol = homogeneous_part.shape

        additional_row = len(additional_pair)
        # additional relationship
        new_homogeneous = np.zeros(
            (nrow + additional_row, ncol + additional_row), dtype=homogeneous_part.dtype
        )
        new_homogeneous[0:nrow, 0:ncol] = homogeneous_part
            
        additional_indices = [ considered_ao_pairs.index(ap) for ap in additional_pair ]
        for i, equivalent_index in enumerate(additional_indices):
            new_homogeneous[nrow + i, equivalent_index] = 1.0
            new_homogeneous[nrow + i, ncol + i] = -1.0

        return new_homogeneous
        
    
    def __init__(self, subspace_interactions: typing.List[InteractingAOSubspace]) -> None:
        import warnings
        warnings.warn(f"{self.__name__} should not be used!")

        all_AOpairs: typing.List[AOPair] = []
        for subspace in subspace_interactions:
            all_AOpairs += subspace.all_AOpairs
        
        considered_ao_pairs: typing.List[AOPair] = []
        for subspace in subspace_interactions:
            for aop in subspace.all_AOpairs:
                if aop not in considered_ao_pairs:
                    considered_ao_pairs.append(aop)

        additional_pair: typing.List[AOPair] = []
        for pair in all_AOpairs:
            found = False
            for considered in considered_ao_pairs:
                if pair.strict_match(considered): 
                    found = True
                    break
            if not found:
                additional_pair.append(pair)

        self._all_AOpairs = considered_ao_pairs + additional_pair
        homogeneous_matrix = self.construct_homogeneous_part(
            subspace_interactions, considered_ao_pairs, additional_pair
        )
        self._homogeneous_equation = LinearEquation(homogeneous_matrix)


    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_AOpairs

    @property
    def homogeneous_equation(self) -> LinearEquation:
        return self._homogeneous_equation


class CombinedEquivalentInteractingAO(InteractionBase):
    """Solving crystal interaction using symmetry

    this class use the symmetry relationship to recover the complete matrix for a set of equivalent atoms (not the whole structure). We can build the full matrix for the whole structure further by the BlockInteraction class.
    """
    def __init__(self, 
        original_cluster: CenteredCluster,
        original_interactingAO: InteractingAOSubspace,
        generated_cluster: typing.List[CenteredCluster],
        possible_rotations: typing.List[np.ndarray]
    ) -> None:

        self.original_cluster = original_cluster
        self.original_interaction = original_interactingAO
        self.generated_clusters = generated_cluster

        if len(self.generated_clusters) == 0:
            self.generating_operations = []
            return

        self.generating_operations = []

        all_aopairs_by_cluster = []
        all_other_aopairs = []
        for cluster in self.generated_clusters:
            all_aopairs_on_cluster = []
            center_nls = _get_orbital_ln_from_string(cluster.center_site.orbitals)
            center_vs = VectorSpace.from_sites_and_orbitals(
                [cluster.center_site.site], OrbitalsList([Orbitals(center_nls)])
            )

            neighbor_nls = _get_orbital_ln_from_string(cluster.neighbor_sites[0].orbitals)
            neighbor_sites = \
                [cluster.center_site.site] + [csite.site for csite in cluster.neighbor_sites]
            neighbor_vs = VectorSpace.from_sites_and_orbitals(
                neighbor_sites, 
                OrbitalsList([Orbitals(center_nls)] + [Orbitals(neighbor_nls)]*len(neighbor_sites))
            )

            center_aos = _get_AO_from_CrystalSites_OrbitalList(
                [cluster.center_site], center_vs.orbital_list
            )
            neighbor_aos = _get_AO_from_CrystalSites_OrbitalList(
                [cluster.center_site] + cluster.neighbor_sites, neighbor_vs.orbital_list
            ) 

            for cao in center_aos:
                for nao in neighbor_aos:
                    pair = AOPair(cao, nao)
                    all_aopairs_on_cluster.append(pair)
                    all_other_aopairs.append(pair)
            
            all_aopairs_by_cluster.append(all_aopairs_on_cluster)
    
        self._all_AOpairs = self.original_interaction.all_AOpairs + all_other_aopairs

        if len(self._all_AOpairs) != \
            (len(generated_cluster) + 1) * len(self.original_interaction.all_AOpairs):
            raise RuntimeError("CombinedEquivalentInteractingAO " \
                + "something is wrong, maybe the aotms are not equivalent")

        known_number_of_homogeneous = \
            self.original_interaction.homogeneous_matrix.shape[0]

        new_homogeneous = np.zeros(
            (
                known_number_of_homogeneous + len(all_other_aopairs), 
                len(self._all_AOpairs)
            ), 
            dtype=self.original_interaction.homogeneous_matrix.dtype
        )
        
        new_homogeneous[
            0:known_number_of_homogeneous, 
            0:len(self.original_interaction.all_AOpairs)
        ] = self.original_interaction.homogeneous_matrix
        
        rotater = AOpairRotater(self._all_AOpairs)
        counter = known_number_of_homogeneous
        for cluster in self.generated_clusters:
            set1 = self.get_array_representation(self.original_cluster)
            set2 = self.get_array_representation(cluster)
            # find rotation
            rotation = None
            rot44 = np.eye(4,4)
            for rot33 in possible_rotations:
                rot44[1:4, 1:4] = rot33
                rotated_array1 = np.einsum("ij,zj->zi", rot44, set1)
                if self.unordered_list_equivalent(rotated_array1, set2, tolerance_structure):
                    rotation = rot33
                    break
            else:
                raise RuntimeError("rototion cannot be found")
            # find translation
            translation = cluster.center_site.absolute_position \
                        - self.original_cluster.center_site.absolute_position
            self.generating_operations.append((rotation, translation))

            for pair_pos, aopair in enumerate(self.original_interaction.all_AOpairs):
                rotated_coefficient = rotater.rotate_aopair(aopair, rotation, translation)
                rotated_coefficient[pair_pos] -= 1.0
                # if then the rotation is identity, we will obtain a zero vector
                new_homogeneous[counter, :] = rotated_coefficient
                counter += 1

        additional = self.additional_equivalence(self._all_AOpairs)
        new_homogeneous = np.vstack([new_homogeneous, additional])
        self._homogeneous_equation = LinearEquation(new_homogeneous)

    @property
    def homogeneous_equation(self) -> LinearEquation:
        return self._homogeneous_equation

    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_AOpairs

    def print_log(self) -> None:
        if len(self.generating_operations) == 0:
            print("## Symmetry generation: no equivalent atoms")
            return
        print("## Combined Equivalent Interacting AO, " \
            + f"there are {len(self.generated_clusters)} sites related by symmetry")
        print("### Generated equivalent atoms:")
        count = 1
        for cluster, (rot, trans) in zip(self.generated_clusters, self.generating_operations):
            print(f"  {count:>2d}" "-rot = ", ("{:>6.2f}"*9).format(*rot.flatten()), end="")
            print(" trans. = ({:>7.3f}{:>7.3f}{:>7.3f})".format(*trans))
            print("    >>Cluster centered on {:>2s} ({:>d})".format(
                cluster.center_site.site.chemical_symbol, cluster.center_site.index_pcell+1
            ), " t = ({:>3d}{:>3d}{:>3d})".format(
                int(cluster.center_site.translation[0]), 
                int(cluster.center_site.translation[1]), 
                int(cluster.center_site.translation[2])
            ))
            count += 1
        print("")
        print(f"### total number of AO pairs = {len(self.all_AOpairs)}")

    @staticmethod
    def get_array_representation(cluster: CenteredCluster):
        result = np.empty((len(cluster.neighbor_sites), 4), dtype=float)
        for i, nsite in enumerate(cluster.neighbor_sites):
            result[i,0] = nsite.site.atomic_number
            result[i,1:] = nsite.site.pos
        return result

    @staticmethod
    def unordered_list_equivalent(
        array1: np.ndarray, array2: np.ndarray, 
        eps
    ) -> int:
        array2_copy = [row for row in array2] # because we want to pop

        for row1 in array1:
            found2 = -1
            for i2, row2 in enumerate(array2_copy):
                if np.allclose(row1, row2, atol=eps):
                    found2 = i2
                    break
            else:
                return False
            array2_copy.pop(found2)
        
        return True

    @staticmethod
    def additional_equivalence(ao_pairs: typing.List[AOPair]) -> np.ndarray:
        """"""
        additional_row = []
        npairs = len(ao_pairs)
        for ipair, ao_pair in enumerate(ao_pairs):
            reverse_aopair = AOPair(ao_pair.r_AO, ao_pair.l_AO)
            for jpair in range(ipair+1, npairs):
                if reverse_aopair.strict_match(ao_pairs[jpair]):
                    coefficients = np.zeros(npairs, dtype=complex_coefficient_type)
                    coefficients[ipair] =  1.0
                    coefficients[jpair] = -1.0
                    additional_row.append(coefficients)
                    break
        
        return np.array(additional_row)

