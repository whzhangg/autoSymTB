import typing, dataclasses
import numpy as np

class SpHar(typing.NamedTuple):
    l: int
    m: int

class nSpHar(typing.NamedTuple):
    n: int
    l: int 
    m: int

class NL(typing.NamedTuple):
    n: int
    l: int

class Orbitals:
    """
    orbitals on a single atoms, it should be initialized as a list of tuple 
    [(n1,l1),(n2,l2),...]
    """
    _aoirrep_ref = {0 : "1x0e",  1 : "1x1o", 2 : "1x2e"}
    _ao_symbol = {0: "s", 1: "p", 2: "d", 3: "f"}
    _aolists = {
        0 : [SpHar(0, 0)], 
        1 : [SpHar(1,-1), SpHar(1, 0), SpHar(1, 1)],
        2 : [SpHar(2,-2), SpHar(2,-1), SpHar(2, 0), SpHar(2, 1), SpHar(2, 2)]
    }
    def __init__(self, nl_list: typing.List[typing.Tuple[int,int]]) -> None:

        self._nl_list: typing.List[NL] = [
            NL(*nl) for nl in nl_list
        ]
    
    def __eq__(self, other) -> bool:
        return self._nl_list == other._nl_list

    @property
    def nl_list(self) -> typing.List[typing.Tuple[int,int]]:
        return self._nl_list

    @property
    def as_spd_str(self) -> typing.List[str]:
        return [ self._ao_symbol[nl.l] for nl in self._nl_list ]

    @property # spherical harmonic nlm
    def sh_list(self) -> typing.List[nSpHar]:
        result = []
        for nl in self._nl_list:
            result += [ nSpHar(nl.n, sh.l, sh.m) for sh in self._aolists[nl.l] ]
        return result

    @property
    def irreps_str(self) -> str:
        return " + ".join([ self._aoirrep_ref[nl.l] for nl in self._nl_list])

    @property
    def num_orb(self) -> int:
        return len(self.sh_list)

    @property
    def slice_dict(self) -> typing.Dict[NL, slice]:
        start = 0
        result = {}
        for nl in self._nl_list:
            end = start + len(self._aolists[nl.l])
            result[nl] = slice(start, end)
            start = end
        return result

    def __contains__(self, nl: NL) -> bool:
        return nl in self._nl_list


@dataclasses.dataclass
class OrbitalsList:
    orbital_list: typing.List[Orbitals]

    @property
    def num_orb(self) -> int:
        return len(self.sh_list)

    @property # spherical harmonic lm
    def sh_list(self) -> typing.List[nSpHar]:
        result = []
        for orbital in self.orbital_list:
            result += orbital.sh_list
        return result

    @property
    def irreps_str(self) -> str:
        return " + ".join([ orb.irreps_str for orb in self.orbital_list ])

    @property
    def atomic_slice_dict(self) -> typing.List[slice]:
        """slice of all orbitals corresponding to one atom"""
        start = 0
        result = []
        for orbitals in self.orbital_list:
            end = start + orbitals.num_orb
            result.append(slice(start, end))
            start = end
        return result

    @property
    def subspace_slice_dict(self) -> typing.List[typing.Dict[str, slice]]:
        start = 0
        result = []
        for orbitals in self.orbital_list:
            slice_each_site = {}
            for l, s in orbitals.slice_dict.items():
                end = start + s.stop - s.start
                slice_each_site[l] = slice(start, end)
                start = end
            result.append(slice_each_site)
        return result


@dataclasses.dataclass
class AO:
    cluster_index: int
    primitive_index: int
    translation: np.ndarray
    chemical_symbol: int
    n: int
    l: int
    m: int

    def __eq__(self, o) -> bool:
        return self.primitive_index == o.primitive_index \
            and self.l == o.l and self.m == o.m and self.n == o.n\
            and np.allclose(self.translation, o.translation)

    def __repr__(self) -> str:
        result = f"n={self.primitive_index}({self.chemical_symbol}) n={self.n} l={self.l} m={self.m}"
        trans = " @{:>5.2f}{:>5.2f}{:>5.2f}".format(*self.translation)
        return result + trans
