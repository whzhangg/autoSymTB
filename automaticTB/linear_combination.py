import numpy as np
import dataclasses, typing
from ase.data import chemical_symbols

class SphericalHarmonic(typing.NamedTuple):
    l: int
    m: int

aoirrep = "1x0e + 1x1o + 1x2e"
AOLISTS = [
    SphericalHarmonic(0, 0),
    SphericalHarmonic(1,-1), SphericalHarmonic(1, 0), SphericalHarmonic(1, 1),
    SphericalHarmonic(2,-2), SphericalHarmonic(2,-1), SphericalHarmonic(2, 0), SphericalHarmonic(2, 1), SphericalHarmonic(2, 2)
]  # in the future, we can add more spherical harmonics corresponding to different n


@dataclasses.dataclass
class Site:
    type: int
    pos: np.ndarray

    def __eq__(self, other) -> bool:
        return self.type == other.type and np.allclose(self.pos, other.pos)

    def rotate(self, matrix:np.ndarray):
        newpos = matrix.dot(self.pos)
        return Site(self.type, newpos)

    def __repr__(self) -> str:
        atom = chemical_symbols[self.type]
        return "{:>2s} @ ({:> 6.2f},{:> 6.2f},{:> 6.2f})".format(atom, *self.pos)

class LinearCombination:
    display_tol = 1e-5
    def __init__(self, sites: typing.List[Site], coefficient: np.ndarray, normalize: bool = False) -> None:
        if len(sites) != len(coefficient):
            raise "Number of coefficient different from number of site"
        if coefficient.shape[1] != len(AOLISTS):
            raise f"only {len(AOLISTS)} atomic orbitals are supported"

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
    def is_normalized(self):
        norm = np.linalg.norm(self.coefficients.flatten())
        return np.isclose(norm, 1.0)

    def scale_coefficients(self, factor: float) -> None:
        self._coefficients *= factor

    def normalized(self):
        # -> LinearCombination
        normalized = self._normalize_coefficients(self._coefficients)
        return LinearCombination(self._sites, normalized)

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
            aline += "{:>+6.3f}({:>1d}{:> 2d})".format(coeff, AOLISTS[i].l, AOLISTS[i].m)
        return aline

    def __str__(self):
        result = ""
        count = 0
        for site, coeff_on_site in zip(self._sites, self._coefficients):
            if np.linalg.norm(coeff_on_site) < self.display_tol: continue
            aline = self._site_string(site, coeff_on_site)
            count += 1
            if count < len(self.sites):
                result += f"{aline}\n"
            elif count == len(self.sites):
                result += f"{aline}"
        return result

    @property
    def pretty_str(self):
        prefix_middle = '├─'
        prefix_last = '└─'
        result = ""
        count = 0
        for site, coeff_on_site in zip(self._sites, self._coefficients):
            if np.linalg.norm(coeff_on_site) < self.display_tol: continue
            aline = self._site_string(site, coeff_on_site)
            count += 1
            if count < len(self.sites):
                result += prefix_middle + f"{aline}\n"
            elif count == len(self.sites):
                result += prefix_last + f"{aline}"
        return result
        
    def __bool__(self):
        return np.linalg.norm(self._coefficients.flatten()) > self.display_tol
