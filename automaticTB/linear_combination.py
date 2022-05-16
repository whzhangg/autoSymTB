import numpy as np
import dataclasses, typing, abc
from ase.data import chemical_symbols
from .structure.sites import Site

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
    def orbitals(self) -> Orbitals:
        return self._orbitals
        
    @property
    def norm(self) -> float:
        return np.linalg.norm(self._coefficients)

    @property
    def is_normalized(self) -> bool:
        return np.isclose(self.norm, 1.0)

    def __bool__(self):
        return self.norm > self.tolorance

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
        start = 0
        for site, orbital in zip(self._sites, self._orbitals):
            end = start + orbital.num_orb
            sliced_coefficient = self._coefficients[start: end]
            if np.linalg.norm(sliced_coefficient) < self.tolorance: continue

            to_display.append(
                self._site_string(site, orbital, sliced_coefficient)
            )

            start = end

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


class LinearCombination:
    display_tol = 1e-5

    def __init__(self, sites: typing.List[Site], orbitals: Orbitals, coefficient: np.ndarray, normalize: bool = False) -> None:
        self._orbitals = orbitals
        if len(sites) != len(coefficient):
            raise "Number of coefficient different from number of site"
        if coefficient.shape[1] != self._orbitals.num_orb:
            raise f"only {self._orbitals.num_orb} atomic orbitals are supported"

        if normalize:
            self._coefficients = self._normalize_coefficients(coefficient)
        else:
            self._coefficients = coefficient
        
        self._sites = sites

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @property
    def sites(self) -> typing.List[Site]:
        return self._sites

    @property
    def orbitals(self) -> Orbitals:
        return self._orbitals

    @property
    def is_normalized(self):
        norm = np.linalg.norm(self.coefficients.flatten())
        return np.isclose(norm, 1.0)

    def scale_coefficients(self, factor: float) -> None:
        self._coefficients *= factor

    def normalized(self): # -> LinearCombination
        normalized = self._normalize_coefficients(self._coefficients)
        return LinearCombination(self._sites, self._orbitals, normalized)

    def get_orbital_subspace(self, orbital: Orbitals): # -> LinearCombination
        coefficient_slice = [
            self._orbitals.slice_dict[o] for o in orbital.orblist
        ]
        sliced = np.hstack([
            self._coefficients[:,s] for s in coefficient_slice
        ])
        return LinearCombination(
            self.sites, orbital, sliced
        )

    def __str__(self):
        prefix_middle = '├─'
        prefix_last = '└─'
        to_display: typing.List[str] = []
        for site, coeff_on_site in zip(self._sites, self._coefficients):
            if np.linalg.norm(coeff_on_site) < self.display_tol: continue
            aline = self._site_string(site, coeff_on_site)
            to_display.append(aline)

        result = ""
        for i, aline in enumerate(to_display):
            if i < len(to_display) - 1:
                result += f"{prefix_middle} {aline}\n"
            else:
                result += f"{prefix_last} {aline}"
        return result
        
    def __bool__(self):
        if np.linalg.norm(self._coefficients.flatten()) > self.display_tol:
            return True
        else:
            return False

    def _normalize_coefficients(self, coefficient: np.ndarray):
        # so that norm equal to 1
        norm = np.linalg.norm(coefficient.flatten())
        if np.isclose(norm, 0.0):
            return coefficient
        else:
            return coefficient / norm

    def _site_string(self, site: Site, coefficient: np.ndarray) -> str:
        aline = str(site) + " :: "
        for i, coeff in enumerate(coefficient):
            if np.abs(coeff) < self.display_tol: continue
            aline += "{:>+6.3f}({:>1d}{:> 2d})".format(
                coeff, self._orbitals.aolist[i].l, self._orbitals.aolist[i].m)
        return aline
