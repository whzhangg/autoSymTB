import typing, dataclasses
import numpy as np

__all__ = ["nSpHar", "NL", "Orbitals", "OrbitalsList", "AO"]


class _SpHar(typing.NamedTuple):
    l: int
    m: int


class nSpHar(typing.NamedTuple):
    """Spherical Harmonics with Orbital number n, specifies an atomic orbital"""
    n: int
    l: int 
    m: int


class NL(typing.NamedTuple):
    """n and l, determines the irreducible representation"""
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
        # this is used to generate a list of nSpHars from a list of NLs
        0 : [_SpHar(0, 0)], 
        1 : [_SpHar(1,-1), _SpHar(1, 0), _SpHar(1, 1)],
        2 : [_SpHar(2,-2), _SpHar(2,-1), _SpHar(2, 0), _SpHar(2, 1), _SpHar(2, 2)]
    }

    def __init__(self, nl_list: typing.List[typing.Tuple[int,int]]) -> None:
        """
        initialize with a list of (n,l) pairs
        """
        self._nl_list: typing.List[NL] = [
            NL(*nl) for nl in nl_list
        ]


    @property
    def nl_list(self) -> typing.List[typing.Tuple[int,int]]:
        """return the list of NL"""
        return self._nl_list

    @property
    def sh_list(self) -> typing.List[nSpHar]:
        """
        return a list of nSpHar
        """
        result = []
        for nl in self._nl_list:
            result += [ nSpHar(nl.n, sh.l, sh.m) for sh in self._aolists[nl.l] ]
        return result

    @property
    def irreps_str(self) -> str:
        """
        generate the string used by the e3nn package
        """
        return " + ".join([ self._aoirrep_ref[nl.l] for nl in self._nl_list])

    @property
    def num_orb(self) -> int:
        return len(self.sh_list)

    @property
    def slice_dict(self) -> typing.Dict[NL, slice]:
        """return a slice for each of the NL"""
        start = 0
        result = {}
        for nl in self._nl_list:
            end = start + len(self._aolists[nl.l])
            result[nl] = slice(start, end)
            start = end
        return result


    def __eq__(self, other) -> bool:
        return self._nl_list == other._nl_list


    def as_spd_str(self) -> typing.List[str]:
        """
        return a list of str: ["1s", "1p", "2s"]
        """
        return [ f"{nl.n}{self._ao_symbol[nl.l]}" for nl in self._nl_list ]


    def __contains__(self, nl: NL) -> bool:
        return nl in self._nl_list


@dataclasses.dataclass
class OrbitalsList:
    """
    A collection of Orbitals object with method mainly combine the output of each orbtial
    """
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
    def subspace_slice_dict(self) -> typing.List[typing.Dict[NL, slice]]:
        """for each atom, provide a list of dictionary from NL to the slice in the list"""
        start = 0
        result: typing.List[typing.Dict[NL, slice]] = []
        for orbitals in self.orbital_list:
            slice_each_site = {}
            for nl, s in orbitals.slice_dict.items():
                end = start + s.stop - s.start
                slice_each_site[nl] = slice(start, end)
                start = end
            result.append(slice_each_site)
        return result


@dataclasses.dataclass
class AO:
    """defines an atomic orbital in a molecular"""
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
