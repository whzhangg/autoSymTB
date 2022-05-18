import typing

class SphericalHarmonic(typing.NamedTuple):
    l: int
    m: int


class Orbitals:
    _aoirrep_ref = {"s": "1x0e",  "p": "1x1o", "d": "1x2e"}
    _aolists = {
        "s": [SphericalHarmonic(0, 0)], 
        "p": [SphericalHarmonic(1,-1), SphericalHarmonic(1, 0), SphericalHarmonic(1, 1)],
        "d": [SphericalHarmonic(2,-2), SphericalHarmonic(2,-1), SphericalHarmonic(2, 0), SphericalHarmonic(2, 1), SphericalHarmonic(2, 2)]
    }
    def __init__(self, orbitals: str) -> None:
        self._orbitals: typing.List[str] = orbitals.split()  # e.g. ["s", "p", "d"]
    
    def __eq__(self, other) -> bool:
        return self._orbitals == other._orbitals

    @property
    def orblist(self) -> typing.List[str]:
        return self._orbitals

    @property
    def aolist(self) -> typing.List[SphericalHarmonic]:
        result = []
        for orb in self._orbitals:
            result += self._aolists[orb]
        return result

    @property
    def irreps_str(self) -> str:
        return " + ".join([ self._aoirrep_ref[o] for o in self._orbitals])

    @property
    def num_orb(self) -> int:
        return sum([len(self._aolists[o]) for o in self._orbitals])

    @property
    def slice_dict(self) -> typing.Dict[str, slice]:
        start = 0
        result = {}
        for orb in self._orbitals:
            end = start + len(self._aolists[orb])
            result[orb] = slice(start, end)
            start = end
        return result

    def __contains__(self, item: str) -> bool:
        return item in self._orbitals

