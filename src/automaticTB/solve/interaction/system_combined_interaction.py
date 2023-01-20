"""
This script is an experimental self-containing solver for the interactions, 
given the so called AOsubspace pair
"""
import typing, dataclasses, collections, copy
import numpy as np

from automaticTB.tools import LinearEquation
from automaticTB.parameters import tolerance_structure, complex_coefficient_type, zero_tolerance
from automaticTB.solve.structure import CenteredCluster
from automaticTB.solve.SALCs import VectorSpace
from automaticTB.solve.atomic_orbitals import OrbitalsList, Orbitals
from automaticTB.solve.rotation import orbital_rotation_from_symmetry_matrix
from .interface import _get_orbital_ln_from_string, _get_AO_from_CrystalSites_OrbitalList
from .interaction_pairs import AO, AOPair
from .interaction_base import InteractionBase, InteractingAOSubspace


@dataclasses.dataclass(eq=True)
class Site:
    pindex: int
    eqindex: int
    t1: int 
    t2: int
    t3: int
    abs_pos: np.ndarray
    symbol: str

    @property
    def rounded_position(self) -> typing.Tuple[float, float,float]:
        digit = np.log10(1 / zero_tolerance)
        return (
            round(self.abs_pos[0], digit),
            round(self.abs_pos[1], digit),
            round(self.abs_pos[2], digit)
        )


    @classmethod
    def from_AO(cls, ao: AO) -> "Site":
        return cls(
            pindex = ao.primitive_index,
            eqindex = ao.equivalent_index,
            t1 = int(ao.translation[0]),
            t2 = int(ao.translation[1]),
            t3 = int(ao.translation[2]),
            abs_pos = ao.absolute_position,
            symbol = ao.chemical_symbol
        )

    @classmethod
    def from_cluster(cls, cluster: CenteredCluster) -> "Site":
        center = cluster.center_site
        return cls(
            pindex = center.index_pcell,
            eqindex = center.equivalent_index,
            t1 = int(center.translation[0]),
            t2 = int(center.translation[1]),
            t3 = int(center.translation[2]),
            abs_pos = center.absolute_position,
            symbol = center.site.chemical_symbol
        )

    def as_tuple(self) -> tuple:
        return (self.pindex, self.t1, self.t2, self.t3)

    def __eq__(self, o: "Site") -> bool:
        return self.as_tuple() == o.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())


def get_translated_AO(aopair: AOPair) -> AOPair:
    l_ao = copy.deepcopy(aopair.l_AO)
    r_ao = copy.deepcopy(aopair.r_AO)
    r_ao.translation -= l_ao.translation
    l_ao.translation = np.zeros(3)
    return AOPair(l_ao, r_ao)


class AOpairRotater:
    """Rotate an AO, return rotated coefficient under `all_ao_pairs` input
    
    it returns complex coefficient of rotated AO
    """
    def __init__(self, 
        all_ao_pairs: typing.List[AOPair]
    ) -> None:
        self.num_aopairs = len(all_ao_pairs)
        self.all_ao_pairs_index = {aopair: ipair for ipair, aopair in enumerate(all_ao_pairs)}
        unique_sites = set()
        for aopair in all_ao_pairs:
            unique_sites.add(Site.from_AO(aopair.l_AO))
            unique_sites.add(Site.from_AO(aopair.r_AO))
        self.unique_sites: typing.List[Site] = list(unique_sites)
        
    def rotate_with_fractional_translation(
        self, ao_pair: AOPair, rotation: np.ndarray, translation: np.ndarray, 
        print_debug: bool = False
    ) -> np.ndarray:
        if print_debug:
            print("rot: ", ("{:>6.2f}"*9).format(*rotation.flatten()))
            print("> input AOPair" + str(ao_pair))

        l_ao = ao_pair.l_AO
        r_ao = ao_pair.r_AO
        
        new_l_absolute_position = l_ao.absolute_position + translation
        dr = r_ao.absolute_position - l_ao.absolute_position
        new_r_absolute_position = np.dot(rotation, dr) + new_l_absolute_position

        new_l_site = None
        for upos in self.unique_sites:
            if np.allclose(new_l_absolute_position, upos.abs_pos, atol = tolerance_structure):
                new_l_site = upos
                break
        else:
            raise RuntimeError("AOpairRotator l: cannot identify the rotated positions")

        new_r_site = None
        for upos in self.unique_sites:
            if np.allclose(new_r_absolute_position, upos.abs_pos, atol=tolerance_structure):
                new_r_site = upos
                break
        else:
            raise RuntimeError("AOpairRotator r: cannot identify the rotated positions")
        
        l_rotated_results = self._rotate_ao_simple(l_ao, rotation)
        r_rotated_results = self._rotate_ao_simple(r_ao, rotation)

        result_coefficients = np.zeros(self.num_aopairs, dtype=complex_coefficient_type)
        reversecoefficients = np.zeros(self.num_aopairs, dtype=complex_coefficient_type)

        result_coefficients[self.all_ao_pairs_index[ao_pair]] -= 1.0
        reversecoefficients[self.all_ao_pairs_index[ao_pair]] -= 1.0

        for ln,ll,lm,lcoe in l_rotated_results:
            if np.abs(lcoe) < zero_tolerance: continue
            l_ao = AO(
                equivalent_index=new_l_site.eqindex,
                primitive_index=new_l_site.pindex,
                absolute_position=new_l_site.abs_pos,
                translation=np.array([new_l_site.t1, new_l_site.t2, new_l_site.t3]),
                chemical_symbol=new_l_site.symbol,
                n = ln, l = ll, m = lm
            )
            for rn,rl,rm,rcoe in r_rotated_results:
                if np.abs(rcoe) < zero_tolerance: continue
                r_ao = AO(
                    equivalent_index=new_r_site.eqindex,
                    primitive_index=new_r_site.pindex,
                    absolute_position=new_r_site.abs_pos,
                    translation=np.array([new_r_site.t1, new_r_site.t2, new_r_site.t3]),
                    chemical_symbol=new_r_site.symbol,
                    n = rn, l = rl, m = rm
                )

                lr_pair = AOPair(l_ao, r_ao)
                rl_pair = get_translated_AO(AOPair(r_ao, l_ao))
                ipair = self.all_ao_pairs_index.get(lr_pair, -1)
                ipair_reversed = self.all_ao_pairs_index.get(rl_pair, -1)

                if print_debug:
                    print("> equivalent pair 1: " + str(lr_pair) + f" coeff = {np.conjugate(lcoe) * rcoe:>.3f}")
                    print("> equivalent pair 2: " + str(rl_pair) + f" coeff = {np.conjugate(lcoe) * rcoe:>.3f}")
                if ipair == -1 or ipair_reversed == -1:
                    raise RuntimeError("rotated pairs not found")
                
                result_coefficients[ipair] += np.conjugate(lcoe) * rcoe
                reversecoefficients[ipair_reversed] += np.conjugate(lcoe) * rcoe
                # maybe we should use conjugate
        
        zeros = np.zeros_like(result_coefficients)
        resulting_pairs = []
        if not np.allclose(result_coefficients, zeros, atol=zero_tolerance):
            resulting_pairs.append(result_coefficients)
        if not np.allclose(reversecoefficients, zeros, atol=zero_tolerance):
            resulting_pairs.append(reversecoefficients)
        return resulting_pairs

    def rotate_aopair(
        self, ao_pair: AOPair, rotation: np.ndarray, translation: np.ndarray, 
        print_debug: bool = False
    ) -> np.ndarray:
        if print_debug:
            print("rot: ", ("{:>6.2f}"*9).format(*rotation.flatten()))
            print("> input AOPair" + str(ao_pair))
        l_ao = ao_pair.l_AO
        r_ao = ao_pair.r_AO
        
        new_l_absolute_position = l_ao.absolute_position + translation
        dr = r_ao.absolute_position - l_ao.absolute_position
        new_r_absolute_position = np.dot(rotation, dr) + new_l_absolute_position

        new_l_site = None
        for upos in self.unique_sites:
            if np.allclose(new_l_absolute_position, upos.abs_pos, atol = tolerance_structure):
                new_l_site = upos
                break
        else:
            raise RuntimeError("AOpairRotator l: cannot identify the rotated positions")

        new_r_site = None
        for upos in self.unique_sites:
            if np.allclose(new_r_absolute_position, upos.abs_pos, atol=tolerance_structure):
                new_r_site = upos
                break
        else:
            return np.zeros(self.num_aopairs, dtype=complex_coefficient_type)
        
        l_rotated_results = self._rotate_ao_simple(l_ao, rotation)
        r_rotated_results = self._rotate_ao_simple(r_ao, rotation)

        result_coefficients = np.zeros(self.num_aopairs, dtype=complex_coefficient_type)

        for ln,ll,lm,lcoe in l_rotated_results:
            if np.abs(lcoe) < zero_tolerance: continue
            l_ao = AO(
                equivalent_index=new_l_site.eqindex,
                primitive_index=new_l_site.pindex,
                absolute_position=new_l_site.abs_pos,
                translation=np.array([new_l_site.t1, new_l_site.t2, new_l_site.t3]),
                chemical_symbol=new_l_site.symbol,
                n = ln, l = ll, m = lm
            )
            for rn,rl,rm,rcoe in r_rotated_results:
                if np.abs(rcoe) < zero_tolerance: continue
                r_ao = AO(
                    equivalent_index=new_r_site.eqindex,
                    primitive_index=new_r_site.pindex,
                    absolute_position=new_r_site.abs_pos,
                    translation=np.array([new_r_site.t1, new_r_site.t2, new_r_site.t3]),
                    chemical_symbol=new_r_site.symbol,
                    n = rn, l = rl, m = rm
                )
                
                ipair = self.all_ao_pairs_index.get(AOPair(l_ao, r_ao), -1)
                if ipair != -1:
                    result_coefficients[ipair] = np.conjugate(lcoe) * rcoe
                
        return result_coefficients

    @staticmethod
    def _rotate_ao_simple(ao, cartesian_rotation):
        orbitals = Orbitals([(ao.n,ao.l)])
        coefficient = np.zeros(len(orbitals.sh_list), dtype = complex_coefficient_type)
        for i, sh in enumerate(orbitals.sh_list):
            if sh.n == ao.n and sh.l == ao.l and sh.m == ao.m:
                coefficient[i] = 1.0
                break
        
        orb_rotation = orbital_rotation_from_symmetry_matrix(
            cartesian_rotation, orbitals.irreps_str
        )
        rotated_coefficient = np.dot(orb_rotation, coefficient)

        result = []
        for i, sh in enumerate(orbitals.sh_list):
            result.append((sh.n, sh.l, sh.m, rotated_coefficient[i]))
        return result


def translate_pair_to_unitcell(aopair: AOPair):
    """simply shift the aopair so that the left ao is in the unit cell"""
    aopair.r_AO

def reverse_translator(all_ao_pairs: typing.List[AOPair]) -> np.ndarray:
    num_aopairs = len(all_ao_pairs)
    all_ao_pairs_index = {aopair: ipair for ipair, aopair in enumerate(all_ao_pairs)}
    all_sites = set()
    for aopair in all_ao_pairs:
        all_sites.add(Site.from_AO(aopair.l_AO))
        all_sites.add(Site.from_AO(aopair.r_AO))
    all_sites: typing.List[Site] = list(all_sites)
    index_translation = {}

    for site in all_sites:
        if site.t1 != 0 or site.t2 != 0 or site.t3 != 0:
            translation = np.array((site.t1, site.t2, site.t3))
            index_translation.setdefault(site.pindex, []).append(translation)
    print(index_translation)
    rows = []
    for ipair, aopair in enumerate(all_ao_pairs):
        if aopair.l_AO.primitive_index != aopair.r_AO.primitive_index \
            or np.allclose(aopair.r_AO.translation, np.zeros(3)): continue
        coefficient = np.zeros(num_aopairs, dtype=complex_coefficient_type)
        coefficient[ipair] = 1.0
        l_ao = aopair.l_AO
        r_ao = aopair.r_AO
        for translation in index_translation[aopair.l_AO.primitive_index]:
            l_ao_trans = AO(
                equivalent_index=l_ao.equivalent_index,
                primitive_index=l_ao.primitive_index,
                absolute_position=l_ao.absolute_position,
                translation=l_ao.translation + translation,
                chemical_symbol=l_ao.chemical_symbol,
                n=l_ao.n, l=l_ao.l, m=l_ao.m
            ) # outside unit cell
            r_ao_trans = AO(
                equivalent_index=r_ao.equivalent_index,
                primitive_index=r_ao.primitive_index,
                absolute_position=r_ao.absolute_position,
                translation=r_ao.translation + translation,
                chemical_symbol=r_ao.chemical_symbol,
                n=r_ao.n, l=r_ao.l, m=r_ao.m
            ) # inside or inside unitcell maybe

            l_ao_trans.translation -= r_ao_trans.translation
            r_ao_trans.translation = np.zeros(3)
            
            equivalent_pair = AOPair(r_ao_trans, l_ao_trans)
            coefficient[all_ao_pairs_index[equivalent_pair]] -= 1.0
            if np.linalg.norm(coefficient) > zero_tolerance:
                rows.append(coefficient)

    return np.array(rows)



class AOPairReverseTranslator:
    def __init__(self, 
        all_ao_pairs: typing.List[AOPair]
    ) -> None:
        self.num_aopairs = len(all_ao_pairs)
        self.all_ao_pairs_index = {aopair: ipair for ipair, aopair in enumerate(all_ao_pairs)}

    def get_translation_reverse_equivalence(
        self, pair_in: AOPair, translation: np.ndarray
    ) -> np.ndarray:
        
        coefficient = np.zeros(self.num_aopairs, dtype=complex_coefficient_type)
        coefficient[self.all_ao_pairs_index[pair_in]] += 1.0
        l_ao = pair_in.l_AO # inside unitcell
        r_ao = pair_in.r_AO # outside unitcell

        if not np.allclose(l_ao.translation, np.zeros(3)):
            raise RuntimeError("l_ao is not in unit cell")
        

        l_ao_trans = AO(
            equivalent_index=l_ao.equivalent_index,
            primitive_index=l_ao.primitive_index,
            absolute_position=l_ao.absolute_position,
            translation=l_ao.translation + translation,
            chemical_symbol=l_ao.chemical_symbol,
            n=l_ao.n, l=l_ao.l, m=l_ao.m
        ) # outside unit cell
        r_ao_trans = AO(
            equivalent_index=r_ao.equivalent_index,
            primitive_index=r_ao.primitive_index,
            absolute_position=r_ao.absolute_position,
            translation=r_ao.translation + translation,
            chemical_symbol=r_ao.chemical_symbol,
            n=r_ao.n, l=r_ao.l, m=r_ao.m
        ) # inside or inside unitcell maybe
        
        shift = r_ao_trans.translation

        r_ao_trans.translation -= shift
        l_ao_trans.translation -= shift
        
        equivalent_pair = AOPair(r_ao_trans, l_ao_trans)
        coefficient[self.all_ao_pairs_index[equivalent_pair]] -= 1.0

        return coefficient
        

class CombinedEquivalentInteractingAO(InteractionBase):
    def __init__(self,
        solved_interactions: typing.List[InteractionBase],
        nonequivalent_clustersets: typing.List[typing.List[CenteredCluster]],
        possible_rotations: np.ndarray,
        reverse_equivalence: bool = True,
        debug_print: bool = False
    ) -> None:
        if len(solved_interactions) != len(nonequivalent_clustersets):
            print("CombinedEquivalentInteraction ",
                f"input number solution {len(solved_interactions)} is different from " ,
                f"the number of non-equivalent clusters {len(nonequivalent_clustersets)} !!")
            raise RuntimeError

        all_ao_pairs = []
        for equivalent_clusters in nonequivalent_clustersets:
            for cluster in equivalent_clusters:
                related_pairs = generate_aopair(cluster)
                all_ao_pairs += related_pairs

        self._all_aopairs = all_ao_pairs

        if debug_print:
            print(f"Total number of all aopairs: {len(self._all_aopairs)}")
        self.num_aopairs = len(self._all_aopairs)        
        aopair_index: typing.Dict[AOPair, int] = {
            ao: iao for iao, ao in enumerate(self._all_aopairs)
        }

        # generate the known homogeneous matrix by stacking them
        self._known_homogeneous_matrix: typing.List[np.ndarray] = []
        for interaction in solved_interactions:
            _block = np.zeros(
                (len(interaction.homogeneous_matrix),self.num_aopairs), 
                dtype=complex_coefficient_type
            )
            print("Known relationship: ", interaction.homogeneous_matrix.shape)
            for ip, pair in enumerate(interaction.all_AOpairs):
                new_index = aopair_index[pair]
                _block[:, new_index] = interaction.homogeneous_matrix[:, ip]
            self._known_homogeneous_matrix.append(np.array(_block))
        
        # generate the rotational equivalence inside the unitcell
        all_sites = set()
        for aopair in self._all_aopairs:
            all_sites.add(Site.from_AO(aopair.l_AO))
            all_sites.add(Site.from_AO(aopair.r_AO))

        self._all_sites: typing.List[Site] = list(all_sites)
        if debug_print:
            print(f"There are total {len(self._all_sites)} sites")
        symmetry_operations = []
        self._symmetry_relationship: typing.List[np.ndarray] = []
        rotator = AOpairRotater(self._all_aopairs)
        """
        for equivalent_clusters in nonequivalent_clustersets:
            set1 = get_array_representation(equivalent_clusters[0])
            operations = []
            for cluster in equivalent_clusters:
                frac_translation = cluster.center_site.absolute_position \
                                - equivalent_clusters[0].center_site.absolute_position
                set2 = get_array_representation(cluster)
                # find rotation
                rotation = None
                rot44 = np.eye(4,4)
                for rot33 in possible_rotations:
                    rot44[1:4, 1:4] = rot33
                    rotated_array1 = np.einsum("ij,zj->zi", rot44, set1)
                    if unordered_list_equivalent(rotated_array1, set2, tolerance_structure):
                        rotation = rot33
                        break
                else:
                    raise RuntimeError("no rotation found")
                
                operations.append((rotation, frac_translation))
                
                from_site = Site.from_cluster(cluster)
                for site in self._all_sites:
                    if site.pindex == from_site.pindex:
                        additional_translation = np.array([
                            site.t1 - from_site.t1, site.t2 - from_site.t2, site.t3 - from_site.t3
                        ])
                        operations.append((rotation, frac_translation + additional_translation))
                
            if debug_print:
                print(f"from site pindex = {equivalent_clusters[0].center_site.index_pcell}")
                for r, t in operations:
                    print(" -rot = ", ("{:>6.2f}"*9).format(*(r.flatten())), end="")
                    print("  trans. = ({:>7.3f}{:>7.3f}{:>7.3f})".format(*t))
        """
        print("")
        print("rotating pairs")

        for aointeractions, equivalent_clusters in zip(
            solved_interactions, nonequivalent_clustersets
        ):
            set1 = get_array_representation(equivalent_clusters[0])

            # each of the cluster in the unit cell
            for cluster in equivalent_clusters:
                frac_translation = cluster.center_site.absolute_position \
                                - equivalent_clusters[0].center_site.absolute_position
                new_rows = []
                set2 = get_array_representation(cluster)
                # find rotation
                rotation = None
                rot44 = np.eye(4,4)
                for rot33 in possible_rotations:
                    rot44[1:4, 1:4] = rot33
                    rotated_array1 = np.einsum("ij,zj->zi", rot44, set1)
                    rotated_array2 = np.einsum('ij,zj->zi', rot44, set2)
                    if unordered_list_equivalent(rotated_array1, set2, tolerance_structure):
                        rotation = rot33
                        break
                else:
                    raise RuntimeError("no rotation found")
                
                for aopair in aointeractions.all_AOpairs:
                    new_rows += rotator.rotate_with_fractional_translation(
                        aopair, rotation, frac_translation, print_debug=False
                    )

                symmetry_operations.append((rotation, frac_translation))
                if debug_print:
                    print("Operation to generate equivalent atoms:")
                    print(" -rot = ", ("{:>6.2f}"*9).format(*(rotation.flatten())), end="")
                    print("  trans. = ({:>7.3f}{:>7.3f}{:>7.3f})".format(*frac_translation))

                relationships = LinearEquation(np.vstack(new_rows))
                print(relationships.row_echelon_form.shape)
                self._symmetry_relationship.append(relationships.row_echelon_form)
        # 

        tmp_linear = LinearEquation(
            np.vstack(self._known_homogeneous_matrix + self._symmetry_relationship)
        )
        #
        self._homogeneous_equation = tmp_linear

    @property
    def homogeneous_equation(self) -> typing.Optional[LinearEquation]:
        return self._homogeneous_equation

    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_aopairs
 

def generate_aopair(cluster: CenteredCluster) -> typing.List[AOPair]:
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

    return all_aopairs_on_cluster 

def get_array_representation(cluster: CenteredCluster):
    result = np.empty((len(cluster.neighbor_sites), 4), dtype=float)
    for i, nsite in enumerate(cluster.neighbor_sites):
        result[i,0] = nsite.site.atomic_number
        result[i,1:] = nsite.site.pos
    return result

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

def run_solve(
    solved_interactions_and_equivalent_clusters: typing.List[
        typing.Tuple[InteractingAOSubspace, typing.List[CenteredCluster]]
    ],
    possible_rotations: np.ndarray
):
    all_aopairs: typing.List[AOPair] = []
    for _, equivalent_clusters in solved_interactions_and_equivalent_clusters:
        for cluster in equivalent_clusters:
            all_aopairs += generate_aopair(cluster)
    
    aopair_index: typing.Dict[AOPair, int] = {}
    for i, aopair in enumerate(all_aopairs):
        aopair_index[aopair] = i

    print(f"total number of aopairs: {len(all_aopairs)}")
    print(f"total number of the set of aopairs: {len(set(all_aopairs))}")

    all_aos = set()
    all_sites = set()
    for aopair in all_aopairs:
        all_aos.add(aopair.l_AO)
        all_aos.add(aopair.r_AO)
        all_sites.add(Site.from_AO(aopair.l_AO))
        all_sites.add(Site.from_AO(aopair.r_AO))

    all_aos = list(all_aos)
    all_sites: typing.List[Site] = list(all_sites)
    print(f"There are total {len(all_aos)} AOs from {len(all_sites)} sites")

    print(f"Generating equivalent relationship")
    new_rows = []
    naopairs = len(all_aopairs)
    for i, aopair in enumerate(all_aopairs):
        reverse_pair = AOPair(aopair.r_AO, aopair.l_AO)
        reverse_pair_index = aopair_index.get(reverse_pair, -1)
        if reverse_pair_index >= 0 and reverse_pair_index > i:
            coefficients = np.zeros(naopairs, dtype=complex_coefficient_type)
            coefficients[i] = 1.0
            coefficients[reverse_pair_index] = -1.0
            new_rows.append(coefficients)
    print(f" -> generated {len(new_rows)} relationship")

    reverse_equivalence = np.array(new_rows)
    # generating all the equivalent sites
    identity_operation = np.eye(3)
    corresponding_operations = []
    rotator = AOpairRotater(all_aopairs)
    sym_equivalences = []
    for aointeractions, equivalent_clusters in solved_interactions_and_equivalent_clusters:
        operations = []

        set1 = get_array_representation(equivalent_clusters[0])
        for cluster in equivalent_clusters:
            set2 = get_array_representation(cluster)
            # find rotation
            rotation = None
            rot44 = np.eye(4,4)
            for rot33 in possible_rotations:
                rot44[1:4, 1:4] = rot33
                rotated_array1 = np.einsum("ij,zj->zi", rot44, set1)
                if unordered_list_equivalent(rotated_array1, set2, tolerance_structure):
                    rotation = rot33
                    break
            else:
                raise RuntimeError("rototion cannot be found")
            # find translation
            frac_translation = cluster.center_site.absolute_position \
                            - equivalent_clusters[0].center_site.absolute_position

            operations.append((rotation, frac_translation))

            for site in all_sites:
                if site.pindex == cluster.center_site.index_pcell and \
                    (site.t1 != 0 or site.t2 != 0 or site.t3 != 0):
                    additional_translation = site.abs_pos - cluster.center_site.absolute_position
                    operations.append((rotation, frac_translation + additional_translation))

        print("Operation to generate equivalent atoms:")
        for iop, op in enumerate(operations):
            print(f"  {iop+1:>2d}", "-rot = ", ("{:>6.2f}"*9).format(*(op[0].flatten())), end="")
            print(" trans. = ({:>7.3f}{:>7.3f}{:>7.3f})".format(*op[1]))

        new_rows = []
        for rot, trans in operations:
            if np.allclose(rot, np.eye(3)) and np.allclose(trans, np.zeros(3)): continue
            for aopair in aointeractions.all_AOpairs:
                rotated_coeff = rotator.rotate_aopair(aopair, rot, trans)
                if np.linalg.norm(rotated_coeff) > zero_tolerance:
                    which_pair = aopair_index[aopair]
                    rotated_coeff[which_pair] -= 1.0
                    new_rows.append(rotated_coeff)
        
        sym_equivalences.append(np.array(new_rows))
        print(f"generated {len(new_rows)} by symmetry operations")


    