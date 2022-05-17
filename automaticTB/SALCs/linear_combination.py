import numpy as np
import dataclasses, typing
from ..structure.sites import Site
from ..rotation import orbital_rotation_from_symmetry_matrix

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


class LinearCombination:
    tolorance = 1e-6

    def __init__(self, sites: typing.List[Site], orbitals: typing.List[Orbitals], coefficient: np.ndarray):
        assert len(sites) == len(orbitals)
        self._sites = sites
        self._orbitals = orbitals
        self._coefficient_length = sum([orb.num_orb for orb in orbitals])
        self._coefficients = coefficient
        assert len(coefficient) == self._coefficient_length

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @property
    def sites(self) -> typing.List[Site]:
        return self._sites

    @property
    def orbitals(self) -> typing.List[Orbitals]:
        return self._orbitals
        
    @property
    def norm(self) -> float:
        return np.linalg.norm(self._coefficients)

    @property
    def is_normalized(self) -> bool:
        return np.isclose(self.norm, 1.0)

    @property
    def nonzero(self) -> bool:
        return self.norm > self.tolorance

    @property
    def atomic_slice(self) -> typing.List[slice]:
        result = []
        start = 0
        for orbital in self._orbitals:
            end = start + orbital.num_orb
            result.append(slice(start, end))
            start = end
        return result

    def __bool__(self):
        return self.nonzero

    def scale_coefficients(self, factor: float) -> None:
        self._coefficients *= factor

    def get_normalized(self): # LinearCombination
        coefficient = self._coefficients
        norm = self.norm
        if norm > self.tolorance:
            coefficient /= norm
        
        return LinearCombination(self._sites, self._orbitals, coefficient)
        
    def __str__(self):
        to_display: typing.List[str] = []
        for site, orbital, coeff_slice in zip(self._sites, self._orbitals, self.atomic_slice):
            sliced_coefficient = self._coefficients[coeff_slice]
            if np.linalg.norm(sliced_coefficient) < self.tolorance: continue

            to_display.append(
                self._site_string(site, orbital, sliced_coefficient)
            )

        prefix_middle = '├─'
        prefix_last = '└─'
        result = ""
        for i, aline in enumerate(to_display):
            if i < len(to_display) - 1:
                result += f"{prefix_middle} {aline}\n"
            else:
                result += f"{prefix_last} {aline}"
        return result
    
    def _site_string(self, site: Site, orbitals: Orbitals, coefficient: np.ndarray) -> str: 
        aline = str(site) + " :: "
        for i, coeff in enumerate(coefficient):
            if np.abs(coeff) < self.tolorance: continue
            aline += "{:>+6.3f}({:>1d}{:> 2d})".format(
                coeff, orbitals.aolist[i].l, orbitals.aolist[i].m)
        return aline

    def general_rotate(self, cartesian_matrix: np.ndarray): # -> LinearCombination
        new_sites, new_coefficient = self._rotate_site_and_coefficient(cartesian_matrix)
        return LinearCombination(
            new_sites, self._orbitals, new_coefficient
        )
        
    def symmetry_rotation(self, cartesian_matrix: np.ndarray): # -> LinearCombination
        new_sites, new_coefficient = self._rotate_site_and_coefficient(cartesian_matrix)
        
        from_which = [
            new_sites.index(site) for site in self._sites
        ]
        ordered_coefficient = np.zeros_like(new_coefficient)
        coeff_slices = self.atomic_slice
        for current_pos, origin_pos in enumerate(from_which):
            ordered_coefficient[coeff_slices[current_pos]] = new_coefficient[coeff_slices[origin_pos]]

        return LinearCombination(
            self._sites, self._orbitals, ordered_coefficient
        )
    
    def _rotate_site_and_coefficient(self, cartesian_matrix) -> typing.Tuple[typing.List[Site], np.ndarray]:
        new_sites = [
            site.rotate(cartesian_matrix) for site in self._sites
        ]
        irreps = " + ".join([orb.irreps_str for orb in self._orbitals])
        orbital_rotataion = orbital_rotation_from_symmetry_matrix(cartesian_matrix, irreps)
        new_coefficient = np.dot(orbital_rotataion, self._coefficients)
        return new_sites, new_coefficient

