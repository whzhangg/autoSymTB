import typing, dataclasses
import numpy as np

class SpHar(typing.NamedTuple):
    l: int
    m: int


class Orbitals:
    _aoirrep_ref = {0 : "1x0e",  1 : "1x1o", 2 : "1x2e"}
    _ao_symbol = {0: "s", 1: "p", 2: "d", 3: "f"}
    _aolists = {
        0 : [SpHar(0, 0)], 
        1 : [SpHar(1,-1), SpHar(1, 0), SpHar(1, 1)],
        2 : [SpHar(2,-2), SpHar(2,-1), SpHar(2, 0), SpHar(2, 1), SpHar(2, 2)]
    }
    def __init__(self, l_list: typing.List[int]) -> None:
        self._l_list = l_list # [0, 1] or [1, 2] etc.
    
    def __eq__(self, other) -> bool:
        return self._l_list == other._l_list

    @property
    def l_list(self) -> typing.List[int]:
        return self._l_list

    @property
    def as_spd_str(self) -> typing.List[str]:
        return [ self._ao_symbol[l] for l in self._l_list ]

    @property # spherical harmonic lm
    def sh_list(self) -> typing.List[SpHar]:
        result = []
        for l in self._l_list:
            result += self._aolists[l]
        return result

    @property
    def irreps_str(self) -> str:
        return " + ".join([ self._aoirrep_ref[l] for l in self._l_list])

    @property
    def num_orb(self) -> int:
        return len(self.sh_list)

    @property
    def slice_dict(self) -> typing.Dict[int, slice]:
        start = 0
        result = {}
        for l in self._l_list:
            end = start + len(self._aolists[l])
            result[l] = slice(start, end)
            start = end
        return result

    def __contains__(self, l: int) -> bool:
        return l in self._l_list


@dataclasses.dataclass
class OrbitalsList:
    orbital_list: typing.List[Orbitals]

    @property
    def num_orb(self) -> int:
        return len(self.sh_list)

    @property # spherical harmonic lm
    def sh_list(self) -> typing.List[SpHar]:
        result = []
        for orbital in self.orbital_list:
            result += orbital.sh_list
        return result

    @property
    def irreps_str(self) -> str:
        return " + ".join([ orb.irreps_str for orb in self.orbital_list ])

    @property
    def atomic_slice_dict(self) -> typing.List[slice]:
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
    l: int
    m: int

    def __eq__(self, o) -> bool:
        return self.primitive_index == o.primitive_index \
            and self.l == o.l and self.m == o.m \
            and np.allclose(self.translation, o.translation)

    def __repr__(self) -> str:
        result = f"n={self.primitive_index}({self.chemical_symbol}) l={self.l} m={self.m}"
        trans = " @{:>5.2f}{:>5.2f}{:>5.2f}".format(*self.translation)
        return result + trans
