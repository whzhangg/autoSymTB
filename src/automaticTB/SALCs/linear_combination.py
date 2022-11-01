import numpy as np
import dataclasses, typing
from ..structure import Site
from ..rotation import orbital_rotation_from_symmetry_matrix
from .symmetrygroup import CartesianOrbitalRotation
from ..atomic_orbitals import Orbitals, OrbitalsList
from ..parameters import zero_tolerance

# it should not be created manually, but from vector space and get_nonzero_LC methods
@dataclasses.dataclass
class LinearCombination:
    sites: typing.List[Site]
    orbital_list: OrbitalsList  # mainly for printing
    coefficients: np.ndarray
        
    @property
    def norm(self) -> float:
        return np.linalg.norm(self.coefficients)

    @property
    def is_normalized(self) -> bool:
        return np.isclose(self.norm, 1.0, atol = zero_tolerance)

    @property
    def nonzero(self) -> bool:
        return self.norm > zero_tolerance

    @property
    def atomic_slice(self) -> typing.List[slice]:
        result = []
        start = 0
        for orbital in self.orbital_list.orbital_list:
            end = start + orbital.num_orb
            result.append(slice(start, end))
            start = end
        return result

    def create_new_with_coefficients(self, new_coefficients: np.ndarray):
        assert len(new_coefficients) == len(self.coefficients)
        assert new_coefficients.dtype == self.coefficients.dtype
        return LinearCombination(self.sites, self.orbital_list, new_coefficients)

    def scale_coefficients(self, factor: float) -> None:
        self.coefficients *= factor

    def get_normalized(self): # LinearCombination
        coefficients = self.coefficients
        norm = self.norm
        if norm > zero_tolerance:
            coefficients /= norm
        return LinearCombination(self.sites, self.orbital_list, coefficients)
       
    def __str__(self):
        to_display: typing.List[str] = []
        for site, orbital, coeff_slice in zip(
            self.sites, self.orbital_list.orbital_list, self.atomic_slice
        ):
            sliced_coefficient = self.coefficients[coeff_slice]
            if np.linalg.norm(sliced_coefficient) < zero_tolerance: continue

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
        lm_name = {
            (0,  0):  "s",
            (1, -1): "py",
            (1,  0): "pz",
            (1,  1): "px",
            (2, -2): "dxy",
            (2, -1): "dyz",
            (2,  0): "dz2",
            (2,  1): "dxz",
            (2,  2): "dx2-y2"
        }
        aline = str(site) + " :: "
        for i, coeff in enumerate(coefficient):
            if np.abs(coeff) < zero_tolerance: continue
            if np.isreal(coeff):
                aline += "{:>+6.3f}".format(coeff.real)
            else:
                aline += "{:>+12.2f}".format(coeff)
            lm = (orbitals.sh_list[i].l, orbitals.sh_list[i].m)
            if lm in lm_name:
                aline += "({:>1d}-{:s})".format(orbitals.sh_list[i].n, lm_name[lm])
            else:
                aline += "({:>1d}{:>2d}{:>2d})".format(
                    orbitals.sh_list[i].n, orbitals.sh_list[i].l, orbitals.sh_list[i].m
                )
        return aline

    def general_rotate(self, cartesian_matrix: np.ndarray): # -> LinearCombination
        new_sites, new_coefficient = self._rotate_site_and_coefficient(cartesian_matrix)
        return LinearCombination(
            new_sites, self.orbital_list, new_coefficient
        )
        
    def symmetry_rotate(self, cartesian_matrix: np.ndarray): # -> LinearCombination
        new_sites, new_coefficient = self._rotate_site_and_coefficient(cartesian_matrix)
        
        from_which = [
            new_sites.index(site) for site in self.sites
        ]
        ordered_coefficient = np.zeros_like(new_coefficient)
        coeff_slices = self.orbital_list.atomic_slice_dict
        for current_pos, origin_pos in enumerate(from_which):
            ordered_coefficient[coeff_slices[current_pos]] = new_coefficient[coeff_slices[origin_pos]]

        return LinearCombination(
            self.sites, self.orbital_list, ordered_coefficient
        )
    
    def _rotate_site_and_coefficient(self, cartesian_matrix) -> typing.Tuple[typing.List[Site], np.ndarray]:
        new_sites = [
            site.rotate(cartesian_matrix) for site in self.sites
        ]
        orbital_rotataion = orbital_rotation_from_symmetry_matrix(cartesian_matrix, self.orbital_list.irreps_str)
        #print(self.orbital_list.irreps_str)
        new_coefficient = np.dot(orbital_rotataion, self.coefficients)
        return new_sites, new_coefficient

    def symmetry_rotate_CartesianOrbital(self, rotation: CartesianOrbitalRotation):
        assert self.orbital_list.irreps_str == rotation.irrep_str
        new_sites = [
            site.rotate(rotation.rot_cartesian) for site in self.sites
        ]
        new_coefficient = np.dot(rotation.rot_orbital, self.coefficients)

        from_which = [
            new_sites.index(site) for site in self.sites
        ]
        ordered_coefficient = np.zeros_like(new_coefficient)
        coeff_slices = self.orbital_list.atomic_slice_dict
        for current_pos, origin_pos in enumerate(from_which):
            ordered_coefficient[coeff_slices[current_pos]] = new_coefficient[coeff_slices[origin_pos]]

        return LinearCombination(
            self.sites, self.orbital_list, ordered_coefficient
        )
