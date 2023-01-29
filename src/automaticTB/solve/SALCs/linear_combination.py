import typing
import dataclasses

#import numba
import numpy as np

from automaticTB import parameters as params
from automaticTB.solve import structure
from automaticTB.solve import rotation
from automaticTB.solve import atomic_orbitals as ao
from .symmetrygroup import CartesianOrbitalRotation
from .orbitals import OrbitalsList
from automaticTB.solve import interaction


def symrotate_orbitalvector(
        lc: "LinearCombination", cart_rot: np.ndarray) -> "LinearCombination":
    """rotation but with reordered site"""
    pass

@dataclasses.dataclass
class OrbitalsVector:
    """a vector in the vector space for fast operation"""
    pos: np.ndarray
    types: np.ndarray
    coefficients: np.ndarray
    irreps_str: str

@dataclasses.dataclass
class LinearCombination:
    """a vector in the """
    pos: np.ndarray
    types: np.ndarray
    coefficients: np.ndarray
    irreps_str: str



@dataclasses.dataclass
class IrrepSymbol:
    symmetry_symbol: str
    main_irrep: str
    main_index: int

    @classmethod
    def from_str(cls, input: str):
        """the main symbol ^ index -> subduction symbol"""
        parts = input.split("->")
        mains = parts[0].split("^")
        main_irrep = mains[0]
        main_index = int(mains[1])
        sym_symbol = "->".join([p.split("^")[0] for p in parts])
        return cls(sym_symbol, main_irrep, main_index)

    def __repr__(self) -> str:
        return f"{self.symmetry_symbol} @ {self.main_index}^th {self.main_irrep}"


class NamedLC(typing.NamedTuple):
    name: IrrepSymbol
    lc: "LinearCombination"

    def __str__(self) -> str:
        return "\n".join([str(self.name), str(self.lc)])


@dataclasses.dataclass
class LinearCombination:
    """A linear combination of atomic orbitals
    
    It should not be created manually, but from vector space or from its
    `get_nonzero_LC` methods
    """
    crysites: typing.List[structure.CrystalSite]
    orbital_list: OrbitalsList
    coefficients: np.ndarray
        
    @property
    def norm(self) -> float:
        return np.linalg.norm(self.coefficients)
    
    
    @property
    def nonzero(self) -> bool:
        return np.any(np.abs(self.coefficients) > params.ztol)


    def create_new_with_coefficients(self, coefficients: np.ndarray) -> "LinearCombination":
        """obtain a new LC with the given coefficients"""
        assert len(coefficients) == len(self.coefficients)
        assert coefficients.dtype == self.coefficients.dtype
        return LinearCombination(self.crysites, self.orbital_list, coefficients)


    def get_normalized(self) -> "LinearCombination":
        """return a normalized LC"""
        coefficients = self.coefficients
        norm = self.norm
        if norm > params.ztol:
            coefficients /= norm
        return LinearCombination(self.crysites, self.orbital_list, coefficients)
       

    def __str__(self) -> str:
        to_display: typing.List[str] = []
        for site, orbital, coeff_slice in zip(
            self.crysites, self.orbital_list.orbital_list, self.orbital_list.atomic_slice_dict
        ):
            sliced_coefficient = self.coefficients[coeff_slice]
            if np.linalg.norm(sliced_coefficient) < params.ztol: continue

            to_display.append(
                _site_string(site, orbital, sliced_coefficient)
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
    
    # the following are rotations

    def general_rotate(self, cartesian_matrix: np.ndarray) -> "LinearCombination":
        new_sites, new_coefficient = self._rotate_site_and_coefficient(cartesian_matrix)
        return LinearCombination(
            new_sites, self.orbital_list, new_coefficient
        )
        
    def symmetry_rotate(self, cartesian_matrix: np.ndarray) -> "LinearCombination":
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
    
    def _rotate_site_and_coefficient(self, cart_rot):
        """rotate sites and coefficients according to cartesian rot."""
        new_sites = [site.rotate(cart_rot) for site in self.sites]
        orbital_rotataion = \
            rotation.orbital_rotation_from_symmetry_matrix(cart_rot, self.orbital_list.irreps_str)
        
        new_coefficient = np.dot(orbital_rotataion, self.coefficients)
        return new_sites, new_coefficient


    def symmetry_rotate_CartesianOrbital(self, rotation: CartesianOrbitalRotation ) \
    -> "LinearCombination":
        """Create new LC from provided orbital rotations
        
        This method assumes that orbitals on each site are the same. 
        Therefore, it is not necessary to reorder the results as in the 
        `.symmetry_rotate` method
        """
        if self.orbital_list.irreps_str != rotation.irrep_str:
            raise RuntimeError(
                f"Rot irrep {rotation.irrep_str} diff from orb. {self.orbital_list.irreps_str}"
            )
        #new_sites = [site.rotate(rotation.rot_cartesian) for site in self.sites]
        new_coefficient = np.dot(rotation.rot_orbital, self.coefficients)
        positions = np.array([[site.pos[0], site.pos[1], site.pos[2]] for site in self.sites])
        from_which = _get_from_which(positions, rotation.rot_cartesian, params.ztol)
        #from_which = [new_sites.index(site) for site in self.sites]
        ordered_coefficient = np.zeros_like(new_coefficient)
        coeff_slices = self.orbital_list.atomic_slice_dict
        for current_pos, origin_pos in enumerate(from_which):
            ordered_coefficient[coeff_slices[current_pos]] \
               = new_coefficient[coeff_slices[origin_pos]]

        #if not np.allclose(new_coefficient, ordered_coefficient, atol=params.ztol):
        #    print("in : ", self.coefficients)
        #    print("new: ", new_coefficient.real)
        #    print("ord: ", ordered_coefficient.real)
        return LinearCombination(
            self.sites, self.orbital_list, ordered_coefficient
        )


def _get_from_which(
    positions: np.ndarray, pos_rotate: np.ndarray, tol: float 
) -> np.ndarray:
    results = np.zeros(len(positions), dtype=np.intc)
    for i0, pos0 in enumerate(positions):
        #pos0 = np.ascontiguousarray(pos0)
        new_pos = np.dot(pos_rotate, pos0)

        for i1, pos1 in enumerate(positions):
            diff = new_pos - pos1
            if diff[0]**2 + diff[1]**2 + diff[2]**2 < tol:
                results[i1] = i0
                break
        else:
            raise RuntimeError("cannot find the rotated atoms")
    return results



def _site_string(site: structure.CrystalSite, orb: ao.Orbitals, coe: np.ndarray) -> str: 
    lm_name = {
        (0,  0):  "s",
        (1, -1): "py", (1,  0): "pz", (1,  1): "px",
        (2, -2): "dxy", (2, -1): "dyz", (2,  0): "dz2", (2,  1): "dxz", (2,  2): "dx2-y2"
    }
    aline = str(site) + " :: "
    for i, coeff in enumerate(coe):
        if np.abs(coeff) < params.ztol: continue
        if np.isreal(coeff):
            aline += "{:>+6.3f}".format(coeff.real)
        else:
            aline += "{:>+12.2f}".format(coeff)
        lm = (orb.sh_list[i].l, orb.sh_list[i].m)
        if lm in lm_name:
            aline += "({:>1d}-{:s})".format(orb.sh_list[i].n, lm_name[lm])
        else:
            aline += "({:>1d}{:>2d}{:>2d})".format(
                orb.sh_list[i].n, orb.sh_list[i].l, orb.sh_list[i].m
            )
    return aline