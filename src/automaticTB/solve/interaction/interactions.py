import typing
import dataclasses

import numpy as np

from automaticTB import tools
from automaticTB import parameters as params
from automaticTB.solve import atomic_orbitals
from automaticTB.solve import structure
from automaticTB.solve import SALCs
from .ao_rotation_tools import AOpairRotater, find_rotation_btw_clusters, get_translated_AO
from .interaction_pairs import (
    AOPair, AOSubspace, get_orbital_ln_from_string,
    generate_aopair_from_cluster, get_AO_from_CrystalSites_OrbitalList,
)


def get_InteractingAOSubspaces_from_cluster(
    cluster: structure.CenteredCluster
) -> typing.List[typing.Tuple[AOSubspace, AOSubspace]]:
    """solved a centered equivalent cluster into a set of interacting subspaces"""

    center_vectorspace: typing.List[SALCs.VectorSpace] = []
    center_namedLCs: typing.List[typing.List[SALCs.NamedLC]] = []
    center_nls = get_orbital_ln_from_string(cluster.center_site.orbitals)
    for cnl in center_nls:
        vs = SALCs.VectorSpace.from_sites_and_orbitals(
                [cluster.center_site.site], 
                atomic_orbitals.OrbitalsList(
                    [atomic_orbitals.Orbitals([cnl])]
                )
            )
        center_vectorspace.append(vs)
        center_namedLCs.append(
            SALCs.decompose_vectorspace_to_namedLC(vs, cluster.sitesymmetrygroup)
        )

    neighbor_vectorspace: typing.List[SALCs.VectorSpace] = []
    neighbor_namedLCs: typing.List[typing.List[SALCs.NamedLC]] = []
    neighbor_nls = get_orbital_ln_from_string(cluster.neighbor_sites[0].orbitals)
    neighborsites = [csite.site for csite in cluster.neighbor_sites]
    for nnl in neighbor_nls:
        vs = SALCs.VectorSpace.from_sites_and_orbitals(
                neighborsites, 
                atomic_orbitals.OrbitalsList(
                    [atomic_orbitals.Orbitals([nnl])] * len(neighborsites)
                )
            )
        neighbor_vectorspace.append(vs)
        neighbor_namedLCs.append(
            SALCs.decompose_vectorspace_to_namedLC(vs, cluster.sitesymmetrygroup)
        )
    
    subspaces_pairs = []
    for left_vs, left_nlc in zip(center_vectorspace, center_namedLCs):
        left_subspace = \
            AOSubspace(
                get_AO_from_CrystalSites_OrbitalList(
                    [cluster.center_site], left_vs.orbital_list
                ),
                left_nlc
            )
            
        for right_vs, right_nlc in zip(neighbor_vectorspace, neighbor_namedLCs):
            right_subspace = \
                AOSubspace(
                    get_AO_from_CrystalSites_OrbitalList(
                        cluster.neighbor_sites, right_vs.orbital_list
                    ),
                    right_nlc
                )
            subspaces_pairs.append((left_subspace, right_subspace))

    return subspaces_pairs


@dataclasses.dataclass
class InteractionSpace:
    """
    free_indices could be empty, in which case homogeneous_equation is full rank
    homoegneous_equation could be None, in which case len(free) is all aopair
    """
    all_AOPairs: typing.List[AOPair]
    free_indices: typing.List[int]
    homogeneous_equation: typing.Optional[np.ndarray]


    def print_free_AOpairs(self) -> None:
        if self.free_indices:
            for i, f in enumerate(self.free_indices):
                print(f"  {i+1:>3d} " + str(self.all_AOPairs[f]))


    def solve_all(self, values: typing.List[float]) -> typing.List[float]:
        all_indices = range(len(self.all_AOPairs))

        if len(values) != len(self.free_indices):
            raise RuntimeError(
                "{:s}: number of input values of interaction {:d} != required {:d}".format(
                    self.__name__, len(values), len(self.free_indices)
                )
            )
    
        if not self.free_indices:
            return np.zeros(len(self.all_AOPairs), dtype=params.COMPLEX_TYPE)
        elif set(self.free_indices) == set(all_indices):
            indices_list = { f:i for i, f in enumerate(self.free_indices)}
            return [values[indices_list[i]] for i in range(len(self.all_AOPairs))]
        
        # obtain values for additional_required_indices
        new_rows = np.zeros(
            (len(self.free_indices), len(self.all_AOPairs)), dtype=params.COMPLEX_TYPE)

        for i, ipair in enumerate(self.free_indices):
            new_rows[i, ipair] = 1.0

        tmp = tools.LinearEquation.from_equation(np.vstack([self.homogeneous_equation, new_rows]))
        # additional required index
        additional_required_indices = tmp.free_variable_indices

        if additional_required_indices:

            related_indices = set()
            indices = np.array(additional_required_indices)
            for row in self.homogeneous_equation:
                if np.all(np.isclose(row[indices], 0.0, atol = params.ztol)):
                    continue
                related_indices |= set(
                    np.argwhere(np.invert(np.isclose(row, 0.0, atol=params.ztol))).flatten()
                )
            # AOs that depend on the selected additional_required_index

            solvable_indices = list(set(range(len(self.all_AOPairs))) - related_indices).sort()
            map_from_old_indices = {old: new for new, old in enumerate(solvable_indices)}

            solvable_parts = tools.LinearEquation.from_equation(
                self.homogeneous_equation[:, np.array(solvable_indices)]
            )
            free_indices_for_solvable_part = [map_from_old_indices[i] for i in self.free_indices]

            solved_values = tools.solve_matrix(solvable_parts, free_indices_for_solvable_part, values)
            solved_aopair_values = {
                self.all_AOPairs[si]:solved_values[i] for i, si in enumerate(solvable_indices)
            } 

            additional_values = []
            for additional_rquired_i in additional_required_indices:
                pair = self.all_AOPairs[additional_rquired_i]
                reverse_pair = get_translated_AO(AOPair(pair.r_AO, pair.l_AO))
                if pair in solved_aopair_values:
                    additional_values.append(solved_aopair_values[pair])
                elif reverse_pair in solved_aopair_values:
                    additional_values.append(np.conjugate(solved_aopair_values[reverse_pair]))
                else:
                    raise RuntimeError("interaction pair not found")

        else:
            additional_values = []

        all_required_indices = self.free_indices + self.additional_required_indices
        all_required_values = np.hstack([values, np.array(additional_values)])

        return tools.solve_matrix(self.homogeneous_equation, all_required_indices, all_required_values)


    def print_log(self) -> None:
        print( "## Atomic Orbital Interactions")
        print(f"  the shape of homogeneous equation: ", self.homogeneous_equation.shape)
        nrow, ncol = self.homogeneous_equation.shape
        print(f"  num. of final required interactions: {ncol - nrow}")
        print(f"  free/total num. interactions: {len(self.free_indices)}/{len(self.all_AOPairs)}")
        if self.free_indices:
            for i, f in enumerate(self.free_indices):
                print(f"  {i+1:>3d} " + str(self.all_AOPairs[f]))
        else:
            raise RuntimeError("free interaction are not obtained? (InteractionBase)")


    @classmethod
    def from_SimpleInteraction(
        cls, allpairs: typing.List[AOPair], matrix: np.ndarray, verbose: bool = False
    ) -> "InteractionSpace":
        if len(matrix) > 0:
            linear = tools.LinearEquation.from_equation(matrix)
            return cls(
                allpairs, linear.free_variable_indices, linear.row_echelon_form
            )
        else:
            return cls(allpairs, list(range(len(allpairs))), None)


    @classmethod
    def from_Blocks(
        cls, interactions: typing.List["InteractionSpace"], verbose:bool = False
    ) -> "InteractionSpace":
        all_aopairs = []

        block_diagonal_size = []
        for interaction in interactions:
            all_aopairs += interaction.all_AOPairs
            homo_nrow = 0
            if interaction.homogeneous_equation is not None:
                homo_nrow = len(interaction.homogeneous_equation)
            block_diagonal_size.append(
                (
                    len(interaction.all_AOPairs), 
                    homo_nrow
                )
            )

        nao = len(all_aopairs)
        nrow_homo = sum(n for _,n in block_diagonal_size)
        homo_matrix = np.zeros((nrow_homo, nao), dtype=params.COMPLEX_TYPE)

        nr1_start = 0 # homogeneous matrix
        col_start = 0
        for (ncol, nr1), interaction in zip(block_diagonal_size, interactions):
                
            nr1_end = nr1_start + nr1
            col_end = col_start + ncol

            if nr1 > 0:
                homo_matrix[nr1_start:nr1_end, col_start:col_end] \
                    = interaction.homogeneous_equation

            col_start = col_end
            nr1_start = nr1_end
            
        return cls.from_SimpleInteraction(
            all_aopairs, homo_matrix
        )


    @classmethod
    def from_AOSubspace(
        cls, l_subspace: AOSubspace, r_subspace: AOSubspace, verbose: bool = False
    ) -> "InteractionSpace":
        """create from two orbital spaces"""
        _forbidden_symbol = 0

        all_aopairs = [ 
            AOPair.from_pair(p) for p in tools.tensor_dot(l_subspace.aos, r_subspace.aos) 
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

        references: typing.List[int] = []
        tps = np.array([tp for _, _, tp in reps_tp_list])
        memory = {}
        symbols = 1
        for rep1, rep2, _ in reps_tp_list:
            if rep1.symmetry_symbol != rep2.symmetry_symbol:
                references.append(_forbidden_symbol)
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
        
        non_homogeneous  = []
        homogeneous_part = [
            tps[i] for i, type in enumerate(references) if type == _forbidden_symbol
        ]
        all_types = set(references) - set([_forbidden_symbol])
        for thetype in all_types:
            indices = [i for i, type in enumerate(references) if type == thetype]
            if len(indices) > 1:
                for i in indices[1:]:
                    homogeneous_part.append(tps[i] - tps[indices[0]])
            non_homogeneous.append(tps[indices[0]])

        if len(homogeneous_part) + len(non_homogeneous) != len(tps):
            raise RuntimeError("_obtain_homogeneous_nonhomogeneous_equation " + \
                "separation of homo-inhomo parts is not successful")
        
        homogeneous_part = np.array(homogeneous_part)
        non_homogeneous = np.array(non_homogeneous)

        if verbose:
            l_str = {0:"s", 1:"p", 2:"d", 3:"f"}
            tmp = l_subspace.aos[0]
            l_orb = f"{tmp.chemical_symbol:>2s}({tmp.n:>2d}{l_str[tmp.l]})"
            tmp = r_subspace.aos[0]
            r_orb = f"{tmp.chemical_symbol:>2s}({tmp.n:>2d}{l_str[tmp.l]})"
            result = cls.from_SimpleInteraction(all_aopairs, homogeneous_part)
            print(
                f"## {l_orb} -> {r_orb} (free/total) orb. interactions: ",
                f"({len(result.free_indices)}/{len(all_aopairs)})"
            )
            result.print_free_AOpairs()
            return result
        else:
            return cls.from_SimpleInteraction(all_aopairs, homogeneous_part)


    @classmethod
    def from_solvedInteraction_and_symmetry(cls, 
        solved_interactions: typing.List["InteractionSpace"],
        nonequivalent_clustersets: typing.List[typing.List[structure.CenteredCluster]],
        possible_rotations: np.ndarray,
        verbose: bool = False
    ) -> "InteractionSpace":
        if len(solved_interactions) != len(nonequivalent_clustersets):
            print("CombinedEquivalentInteraction ",
                f"input number solution {len(solved_interactions)} is different from " ,
                f"the number of non-equivalent clusters {len(nonequivalent_clustersets)} !!")
            raise RuntimeError
        
        all_aopairs = []
        for equivalent_clusters in nonequivalent_clustersets:
            for cluster in equivalent_clusters:
                related_pairs = generate_aopair_from_cluster(cluster)
                all_aopairs += related_pairs

        num_aopairs = len(all_aopairs)        
        aopair_index: typing.Dict[AOPair, int] = {
            ao: iao for iao, ao in enumerate(all_aopairs)
        }

        #print(f"number of all aopairs: {num_aopairs}")
        #for i, pair in enumerate(all_aopairs):
        #    print(f"{i+1:>3d} " + repr(pair))
        #print("")
        # generate the known homogeneous matrix by stacking them
        homogeneous_relationship: typing.List[np.ndarray] = []
        for interaction in solved_interactions:
            _block = np.zeros(
                (len(interaction.homogeneous_equation), num_aopairs), 
                dtype=params.COMPLEX_TYPE
            )
            for ip, pair in enumerate(interaction.all_AOPairs):
                new_index = aopair_index[pair]
                _block[:, new_index] = interaction.homogeneous_equation[:, ip]
            homogeneous_relationship.append(np.array(_block))
        
        # generate the rotational equivalence inside the unitcell
        symmetry_operations = []
        symmetry_relationships: typing.List[np.ndarray] = []
        symmetry_reverse_relationships: typing.List[np.ndarray] = []
        rotator = AOpairRotater(all_aopairs)

        for aointeractions, eq_clusters in zip(solved_interactions, nonequivalent_clustersets):
            # each of the cluster in the unit cell
            operations = []
            for cluster in eq_clusters:
                rotation = find_rotation_btw_clusters(
                    eq_clusters[0], cluster, possible_rotations
                )
                frac_translation = cluster.center_site.absolute_position \
                                - eq_clusters[0].center_site.absolute_position
                operations.append((rotation, frac_translation))
                
                for aopair in aointeractions.all_AOPairs:
                    sym_relation, sym_rev_relation = rotator.rotate(
                        aopair, rotation, frac_translation, print_debug=False
                    )
                    if not np.all(np.isclose(sym_relation, 0.0, atol=params.ztol)):
                        symmetry_relationships.append(sym_relation)
                    if not np.all(np.isclose(sym_rev_relation, 0.0, atol=params.ztol)):
                        symmetry_reverse_relationships.append(sym_rev_relation)
            
            symmetry_operations.append(operations)
                
        homogeneous_equation = tools.LinearEquation.from_equation(np.vstack(
                homogeneous_relationship + \
                symmetry_relationships
            ))

        equation_with_additional_relationship = tools.LinearEquation.from_equation(np.vstack(
            homogeneous_relationship + \
            symmetry_relationships + \
            symmetry_reverse_relationships
        ))

        if verbose:
            print("## Orbital relationship of the whole structure are generated using symmetry...")
            print(f"  there are {len(symmetry_operations)} unique positions in the unit cell")
            for operations, eq_clusters in zip(symmetry_operations, nonequivalent_clustersets):
                first_pindex = eq_clusters[0].center_site.index_pcell + 1
                element = eq_clusters[0].center_site.site.chemical_symbol
                print("")
                print(f"### sites generated from atom with pindex = {first_pindex} ({element:>2s})")
                for op, cluster in zip(operations, eq_clusters):
                    rot, trans = op
                    pindex = cluster.center_site.index_pcell + 1
                    element = cluster.center_site.site.chemical_symbol
                    print(f"    pindex = {pindex:>2d} ({element:>2s})     rot:", end = "  ")
                    print(("{:>8.3f}"*3).format(*rot[0]))
                    print((" "*31 + "{:>8.3f}"*3).format(*rot[1]))
                    print((" "*31 + "{:>8.3f}"*3).format(*rot[2]))
                    print("                 translation: ({:>8.3f}{:>8.3f}{:>8.3f})".format(*trans))
            print("")
            nconstrains = 0
            for hr in homogeneous_relationship:
                nconstrains += len(hr)
            print("  num. of constrains posed by point group symmetry              : ",
                    nconstrains)
            print("  num. of equivalent relationship generated from symmetry op.   : ", 
                    len(symmetry_relationships))
            print("  num. of conjugate relationship <i|H|j> = <j|H|i>* can be used : ", 
                    len(symmetry_reverse_relationships))
            print("  num. of the required parameters to solve all values           : ",
                    len(homogeneous_equation.free_variable_indices))
            print("  num. of the free parameters with conjugate relationships      : ",
                    len(equation_with_additional_relationship.free_variable_indices))
                    

                
        
        return cls(
            all_aopairs, 
            equation_with_additional_relationship.free_variable_indices, 
            homogeneous_equation.row_echelon_form
        )
