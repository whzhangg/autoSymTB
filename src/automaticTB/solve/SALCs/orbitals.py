import re
import typing
import dataclasses


class nSpHar(typing.NamedTuple):
    n: int
    l: int 
    m: int


class Orbitals:
    """
    orbitals on a single atoms
    """
    _aoirrep_ref = {0 : "1x0e",  1 : "1x1o", 2 : "1x2e"}
    _ao_symbol = {0: "s", 1: "p", 2: "d", 3: "f"}
    _aolists = {
        # this is used to generate a list of _nSpHars from a list of _NLs
        0 : [(0, 0)], 
        1 : [(1,-1), (1, 0), (1, 1)],
        2 : [(2,-2), (2,-1), (2, 0), (2, 1), (2, 2)]
    }

    def __init__(self, nl_list: typing.List[typing.Tuple[int,int]]) -> None:
        """initialize with (n,l) list: eg: [(3,0),(3,1)] for 3s3p."""
        # generate irreps
        self.nl_list = nl_list
        self.irreps_str = " + ".join([self._aoirrep_ref[l] for _,l in nl_list])
        # generate sh_list
        self.sh_list: typing.List[nSpHar] = []
        for n, l in nl_list:
            self.sh_list += [nSpHar(n,l,m) for l,m in self._aolists[l]]
    
    @property
    def num_orb(self) -> int:
        return len(self.sh_list)

    @classmethod
    def from_str(cls, orbital: str) -> "Orbitals": 
        """parse orbitals like 3p3d4s -> Orbitals"""
        l_dict = {s:i for i,s in cls._ao_symbol.items()}
        numbers = re.findall(r"\d", orbital)
        symbols = re.findall(r"[spdf]", orbital)

        if len(numbers) != len(symbols):
            raise RuntimeError(f"orbital formula wrong: {orbital}")

        return cls([(int(n),l_dict[lsym]) for n,lsym in zip(numbers, symbols)])
        

    def to_str(self) -> str:
        return "".join([ f"{nl.n}{self._ao_symbol[nl.l]}" for nl in self._nl_list ])


    def __eq__(self, o: "Orbitals") -> bool:
        return self.sh_list == o.sh_list

# should not be used
@dataclasses.dataclass
class OrbitalsList:
    """A collection of Orbitals object with method mainly combine the output of each orbtial"""
    orbital_list: typing.List[Orbitals]

    @property
    def num_orb(self) -> int:
        return sum(o.num_orb for o in self.orbital_list)

    @property # spherical harmonic lm
    def sh_list(self) -> typing.List[nSpHar]:
        result = []
        for orbital in self.orbital_list:
            result += orbital.sh_list
        return result

    @property
    def irreps_str(self) -> str:
        return " + ".join([orb.irreps_str for orb in self.orbital_list])

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

    @classmethod
    def from_str(cls, orb_strings: typing.List[str]) -> "OrbitalsList":
        return cls([Orbitals.from_str(s) for s in orb_strings])


    def to_str(self) -> typing.List[str]:
        return [o.to_str() for o in self.orbital_list]