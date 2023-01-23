import typing, dataclasses, copy, numpy as np

from automaticTB.parameters import complex_coefficient_type, tolerance_structure, zero_tolerance
from automaticTB.solve.atomic_orbitals import Orbitals
from automaticTB.solve.structure import CenteredCluster
from automaticTB.solve.rotation import orbital_rotation_from_symmetry_matrix
from .interaction_pairs import AO, AOPair


def get_translated_AO(aopair: AOPair) -> AOPair:
    """translate the pair so that the left AO is at the center"""
    l_ao = copy.deepcopy(aopair.l_AO)
    r_ao = copy.deepcopy(aopair.r_AO)
    r_ao.translation -= l_ao.translation
    l_ao.translation = np.zeros(3)
    return AOPair(l_ao, r_ao)


def unordered_list_equivalent(array1: np.ndarray, array2: np.ndarray, eps: float) -> bool:
    """return True if the rows in two input array are the same"""
    rows_in_array2 = [row for row in array2] # because we want to pop
    for row1 in array1:
        found2 = -1
        for i2, row2 in enumerate(rows_in_array2):
            if np.allclose(row1, row2, atol=eps):
                found2 = i2
                break
        else:
            return False
        rows_in_array2.pop(found2) 
    return True


def find_rotation_between_two_clusters(
    cluster1: CenteredCluster, cluster2: CenteredCluster, possible_rotations: np.ndarray
) -> np.ndarray:
    """return the rotation between two clusters, raise error if not found"""

    def _get_array_representation(cluster: CenteredCluster):
        """add atomic number as the additional coordinats"""
        result = np.empty((len(cluster.neighbor_sites), 4), dtype=float)
        for i, nsite in enumerate(cluster.neighbor_sites):
            result[i,0] = nsite.site.atomic_number
            result[i,1:] = nsite.site.pos
        return result

    set1 = _get_array_representation(cluster1)
    set2 = _get_array_representation(cluster2)

    rot44 = np.eye(4,4)
    for rot33 in possible_rotations:
        rot44[1:4, 1:4] = rot33
        rotated_array1 = np.einsum("ij,zj->zi", rot44, set1)
        if unordered_list_equivalent(rotated_array1, set2, tolerance_structure):
            return rot33
    else:
        print("=" * 60)
        print("Cluster 1")
        for row in set1:
            print("    {:>5.1f} @ {:>8.3f},{:>8.3f},{:>8.3f}".format(*row))
        print("Cluster 2")
        for row in set2:
            print("    {:>5.1f} @ {:>8.3f},{:>8.3f},{:>8.3f}".format(*row))
        print("=" * 60)
        raise RuntimeError("no rotation found")


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


class AOpairRotater:
    """Rotate an AO, return rotated coefficient under `all_ao_pairs` input
    
    it returns complex coefficient of rotated AO
    """
    def __init__(self, all_ao_pairs: typing.List[AOPair]) -> None:
        self.num_aopairs = len(all_ao_pairs)
        self.all_ao_pairs_index = {aopair: ipair for ipair, aopair in enumerate(all_ao_pairs)}
        unique_sites = set()
        for aopair in all_ao_pairs:
            unique_sites.add(Site.from_AO(aopair.l_AO))
            unique_sites.add(Site.from_AO(aopair.r_AO))
        self.unique_sites: typing.List[Site] = list(unique_sites)
        
    def rotate(
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
        result_coefficients[self.all_ao_pairs_index[ao_pair]] -= 1.0
        reversecoefficients = np.zeros(self.num_aopairs, dtype=complex_coefficient_type)
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
                rl_pair = get_translated_AO(AOPair(r_ao, l_ao)) # reverse pair
                ipair = self.all_ao_pairs_index.get(lr_pair, -1)
                ipair_reversed = self.all_ao_pairs_index.get(rl_pair, -1)

                if print_debug:
                    print("> equivalent pair 1: ", lr_pair, 
                            f" coeff = {np.conjugate(lcoe) * rcoe:>.3f}")
                    print("> equivalent pair 2: ", rl_pair, 
                            f" coeff = {np.conjugate(lcoe) * rcoe:>.3f}")

                if ipair == -1 or ipair_reversed == -1:
                    print("> equivalent pair 1: ", lr_pair, 
                            f" coeff = {np.conjugate(lcoe) * rcoe:>.3f}")
                    print("> equivalent pair 2: ", rl_pair, 
                            f" coeff = {np.conjugate(lcoe) * rcoe:>.3f}")
                    raise RuntimeError("rotated pairs not found")
                
                result_coefficients[ipair] += np.conjugate(lcoe) * rcoe
                reversecoefficients[ipair_reversed] += np.conjugate(lcoe) * rcoe
                # maybe we should use conjugate
        
        return (result_coefficients, reversecoefficients)
        zeros = np.zeros_like(result_coefficients)
        resulting_pairs = []
        if not np.allclose(result_coefficients, zeros, atol=zero_tolerance):
            resulting_pairs.append(result_coefficients)
        if self.use_reverse_equivalence and \
            not np.allclose(reversecoefficients, zeros, atol=zero_tolerance):
            resulting_pairs.append(reversecoefficients)
        return resulting_pairs


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
