import re
import typing
import dataclasses

import numpy as np

from automaticTB import tools
from automaticTB import parameters as params
from automaticTB.solve import SALCs
from automaticTB.solve import structure
from automaticTB.solve import atomic_orbitals

@dataclasses.dataclass
class AO:
    """defines an atomic orbital in a molecular"""
    equivalent_index: int
    primitive_index: int
    absolute_position: np.ndarray
    translation: np.ndarray
    chemical_symbol: str
    n: int
    l: int
    m: int

    @property
    def atomic_number(self) -> int:
        return tools.atomic_numbers[self.chemical_symbol]


    def __eq__(self, o: "AO") -> bool:
        return  self.as_tuple() == o.as_tuple()

    def __repr__(self) -> str:
        result = f"{self.chemical_symbol:>2s} (i_p={self.primitive_index:>2d})"
        result+= f" n ={self.n:>2d}; l ={self.l:>2d}; m ={self.m:>2d}"
        result+=  " @({:>5.1f},{:>5.1f},{:>5.1f})".format(*self.translation)
        return result


    def as_tuple(self) -> tuple:
        t1 = int(self.translation[0])
        t2 = int(self.translation[1])
        t3 = int(self.translation[2])
        return (self.primitive_index, t1, t2, t3, self.n, self.l, self.m)


    def __hash__(self) -> int:
        return hash(
            self.as_tuple()
        )


@dataclasses.dataclass
class AOPair:
    """
    just a pair of AO that enable comparison, AOs can be compared directly
    """
    l_AO: AO
    r_AO: AO

    @property
    def left(self) -> AO:
        return self.l_AO
    
    @property
    def right(self) -> AO:
        return self.r_AO

    @property
    def vector_right_to_left(self) -> np.ndarray:
        return self.l_AO.absolute_position - self.r_AO.absolute_position

    @classmethod
    def from_pair(cls, aopair: tools.Pair) -> "AOPair":
        return cls(aopair.left, aopair.right)

    def as_pair(self) -> tools.Pair:
        return tools.Pair(self.l_AO, self.r_AO)

    def strict_match(self, o: "AOPair") -> bool:
        return  self.l_AO == o.l_AO and self.r_AO == o.r_AO

    def __eq__(self, o: "AOPair") -> bool:
        return self.l_AO == o.l_AO and self.r_AO == o.r_AO
        


    def __str__(self) -> str:
        result = " > Pair: "
        left = self.l_AO; right = self.r_AO
        rij = right.absolute_position - left.absolute_position
        result += f"{left.chemical_symbol:>2s}-{left.primitive_index:0>2d} " 
        result += f"{tools.parse_orbital(left.n, left.l, left.m):>7s} -> "
        result += f"{right.chemical_symbol:>2s}-{right.primitive_index:0>2d} "
        result += f"{tools.parse_orbital(right.n, right.l, right.m):>7s} "
        result += "r = ({:>6.2f},{:>6.2f},{:>6.2f})".format(*rij)
        #result += " t = ({:>3d}{:>3d}{:>3d})".format(*np.array(right.translation, dtype=int))
        return result

    def __hash__(self) -> int:
        return hash(
            self.l_AO.as_tuple() + self.r_AO.as_tuple()
        )


@dataclasses.dataclass
class AOPairWithValue(AOPair):
    """AOPair with value additionally associated with it, overwrite the __eq__ method"""
    value: typing.Union[float, complex]

    def __eq__(self, o: "AOPairWithValue") -> bool:
        pair_OK = super()
        valueOK = np.isclose(self.value, o.value, atol = params.ztol)
        pair_OK = (self.l_AO == o.l_AO and self.r_AO == o.r_AO) or \
                  (self.l_AO == o.r_AO and self.r_AO == o.l_AO)
        
        return  valueOK and pair_OK

    def __str__(self) -> str:
        result = " > Pair: "
        left = self.l_AO; right = self.r_AO
        rij = right.absolute_position - left.absolute_position
        result += f"{left.chemical_symbol:>2s}-{left.primitive_index:0>2d} " 
        result += f"{tools.parse_orbital(left.n, left.l, left.m):>7s} -> "
        result += f"{right.chemical_symbol:>2s}-{right.primitive_index:0>2d} "
        result += f"{tools.parse_orbital(right.n, right.l, right.m):>7s} "
        result += "@ ({:>6.2f},{:>6.2f},{:>6.2f}) ".format(*rij)
        result += f"Hij = {self.value:>.4f}"
        return result

    
@dataclasses.dataclass
class AOSubspace:
    """Defines AOs and a list of their linear combination with different representation
    
    This object is simply a container of a set of named linear combinations with basis 
    specified by aos
    """
    aos: typing.List[AO]
    namedlcs: typing.List[SALCs.NamedLC] # only the components belonging to the AOs

    def __repr__(self) -> str:
        result = ["subspace consists of:"]
        for ao in self.aos:
            result.append(f" {str(ao)}")
        result.append("With linear combinations")
        for nlc in self.namedlcs:
            result.append(f" {str(nlc.name)}")
        return "\n".join(result)



def get_AO_from_CrystalSites_OrbitalList(
    crystalsites: typing.List[structure.CrystalSite],orbitallist: atomic_orbitals.OrbitalsList
) -> typing.List[AO]:
    """given a list of atomic position and orbitals, convernt them to a list of AO"""

    aos = []
    all_sh = orbitallist.sh_list
    i_cluster = 0
    for csite, sh_slice in zip(crystalsites, orbitallist.atomic_slice_dict):
        for sh in all_sh[sh_slice]:
            aos.append(
                AO(
                    equivalent_index = csite.equivalent_index,
                    primitive_index = csite.index_pcell,
                    absolute_position = csite.absolute_position,
                    translation = csite.translation,
                    chemical_symbol = csite.site.chemical_symbol,
                    n = sh.n,
                    l = sh.l,
                    m = sh.m
                )
            )
        i_cluster += 1

    return aos


def get_orbital_ln_from_string(orbital: str) -> typing.List[typing.Tuple[int, int]]:
    """convert string to dict: example "3s3p4s" -> [(3,0),(3,1),(4,0)]"""
    l_dict = {
        "s": 0, "p": 1, "d": 2, "f": 3
    }
    numbers = re.findall(r"\d", orbital)
    symbols = re.findall(r"[spdf]", orbital)
    if len(numbers) != len(symbols):
        print(f"orbital formula wrong: {orbital}")
    return [
        (int(n),l_dict[lsym]) for n,lsym in zip(numbers, symbols)
    ]


def generate_aopair_from_cluster(cluster: structure.CenteredCluster) -> typing.List[AOPair]:
    """generate all AOPair on the given centeredCluster"""
    all_aopairs_on_cluster = []
    center_nls = get_orbital_ln_from_string(cluster.center_site.orbitals)
    center_vs = SALCs.VectorSpace.from_sites_and_orbitals(
                [cluster.center_site.site], atomic_orbitals.OrbitalsList([atomic_orbitals.Orbitals(center_nls)])
            )

    neighbor_nls = [get_orbital_ln_from_string(nsite.orbitals) for nsite in cluster.neighbor_sites]
    neighbor_sites = \
                [cluster.center_site.site] + [csite.site for csite in cluster.neighbor_sites]
    neighbor_vs = SALCs.VectorSpace.from_sites_and_orbitals(
                    neighbor_sites, 
                    atomic_orbitals.OrbitalsList([atomic_orbitals.Orbitals(center_nls)] + [atomic_orbitals.Orbitals(nl) for nl in neighbor_nls])
                )

    center_aos = get_AO_from_CrystalSites_OrbitalList(
                    [cluster.center_site], center_vs.orbital_list
                )
    neighbor_aos = get_AO_from_CrystalSites_OrbitalList(
                    [cluster.center_site] + cluster.neighbor_sites, neighbor_vs.orbital_list
                ) 

    for cao in center_aos:
        for nao in neighbor_aos:
            pair = AOPair(cao, nao)
            all_aopairs_on_cluster.append(pair)

    return all_aopairs_on_cluster 
