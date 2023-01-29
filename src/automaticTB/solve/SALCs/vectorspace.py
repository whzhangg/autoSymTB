import typing
import dataclasses

import numpy as np
from automaticTB import tools
from automaticTB import parameters as params
from automaticTB.solve import rotation
from automaticTB.solve import structure
from automaticTB.solve import interaction
from automaticTB.solve import sitesymmetry as ssym
from .orbitals import Orbitals
from .symmetrygroup import SiteSymmetryGroupwithOrbital


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
    aos: typing.List[interaction.AO]
    coefficients: np.ndarray

    def __str__(self) -> str:
        return "\n".join([str(self.name), str(self.lc)])


@dataclasses.dataclass
class OrbitalsVector:
    """a vector in the vector space for fast operation"""
    tpos4: np.ndarray
    atom_slice: typing.List[slice]
    coefficients: np.ndarray

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.coefficients)

    @property
    def nonzero(self) -> bool:
        return np.any(np.abs(self.coefficients) > params.ztol)

    def create_new_with_coefficients(self, coefficients: np.ndarray) -> "OrbitalsVector":
        """obtain a new LC with the given coefficients"""
        assert len(coefficients) == len(self.coefficients)
        assert coefficients.dtype == self.coefficients.dtype
        return OrbitalsVector(self.tpos4, self.atom_slice, coefficients)


    def get_normalized(self) -> "OrbitalsVector":
        """return a normalized LC"""
        coefficients = self.coefficients
        norm = np.linalg.norm(self.coefficients)
        if norm > params.ztol:
            coefficients /= norm
        return OrbitalsVector(self.tpos4, self.atom_slice, coefficients)


    def __repr__(self) -> str:
        result = [f"Vector composed of {len(self.coefficients)} atomic orbitals"]
        for tp, aslice in zip(self.tpos4, self.atom_slice):
            n = aslice.stop - aslice.start
            type_ = tp[0]
            pos = tp[1:]
            line = f"atomic number {type_:>3d} with {n:>2d} orbitals " 
            line+= "@ ({:>8.3f}{:>8.3f}{:>8.3f}) with coefficients: ".format(*pos)
            line+= ",".join("{:>16.3f}".format(c) for c in self.coefficients[aslice])
            result.append(line)
        return "\n".join(result)


def rotate_orbitalvector(
    orbv: "OrbitalsVector", cart_rot: np.ndarray, orb_rot: np.ndarray
) -> "OrbitalsVector":
    rot44 = np.eye(4)
    rot44[1:,1:] = cart_rot
    rotated_pos = np.einsum("ij,kj->ki", rot44, orbv.tpos4)
    rotated_orbitals = orb_rot @ orbv.coefficients
    return OrbitalsVector(rotated_pos, rotated_orbitals, orbv.atom_slice)


def get_AO_from_CrystalSites_OrbitalList(
    crystalsites: typing.List[structure.CrystalSite], orbitalist: typing.List[Orbitals]
) -> typing.List[interaction.AO]:
    """given a list of atomic position and orbitals, convernt them to a list of AO"""
    aos = []
    for csite, orb in zip(crystalsites, orbitalist):
        for sh in orb.sh_list:
            aos.append(
                interaction.AO(
                    equivalent_index=csite.equivalent_index,
                    primitive_index = csite.index_pcell,
                    absolute_position = csite.absolute_position,
                    translation = csite.translation,
                    chemical_symbol = csite.site.chemical_symbol,
                    n = sh.n,
                    l = sh.l,
                    m = sh.m
                )
            )
    return aos


@dataclasses.dataclass
class OrbitalVectorSpace:
    aos: typing.List[interaction.AO]
    tpos4: np.ndarray
    irreps_str: str
    atom_slices: typing.List[slice]


    @classmethod
    def from_crystalsites_orbitals(
        cls, csites: typing.List[structure.CrystalSite], origin: np.ndrray, 
        orbital_str: typing.List[str]
    ) -> "OrbitalVectorSpace":
        orbital_list: typing.List[Orbitals] = [
                Orbitals.from_str(orb_s) for orb_s in orbital_str]
        irrep_str = " + ".join(orb.irreps_str for orb in orbital_list)
        aos = get_AO_from_CrystalSites_OrbitalList(csites, orbital_list)
        tpos4 = np.array(
            [(cs.atomic_number, *(cs.absolute_position - origin)) for cs in csites]
        )
        atom_slices = []
        start = 0
        for orb in orbital_list:
            end = start + orb.num_orb
            atom_slices.append(slice(start, end))
            start = end

        return cls(
            aos, tpos4, irrep_str, atom_slices
        )


    def get_group_representation(self, group) -> SiteSymmetryGroupwithOrbital:
        return SiteSymmetryGroupwithOrbital.from_sitesymmetrygroup_irreps(group, self.irreps_str)


    def get_a_set_of_basis(self) -> typing.List[np.ndarray]:
        nbasis = len(self.aos)
        return np.eye(nbasis, dtype=params.COMPLEX_TYPE)
        

    def get_orbital_rot_from_cartesian_rot(self, cart_rotation: np.ndarray) -> np.ndarray:
        return rotation.orbital_rotation_from_symmetry_matrix(cart_rotation, self.irreps_str)


    def symrotate_orbitalvector(
        self, coefficients: np.ndarray, cart_rot: np.ndarray, tol: float, 
        orb_rot: typing.Optional[np.ndarray] = None
    ) -> np.ndarray:
        """rotation but with reordered site"""
        rot44 = np.eye(4)
        rot44[1:,1:] = cart_rot
        tpos = self.tpos4

        from_which = np.zeros(len(tpos), dtype=np.intc)
        for i0, pos0 in enumerate(tpos):
            new_pos = np.dot(rot44, pos0)
            for i1, pos1 in enumerate(tpos):
                diff = new_pos - pos1
                if diff[0]**2 + diff[1]**2 + diff[2]**2 < tol:
                    from_which[i1] = i0
                    break
            else:
                raise RuntimeError("cannot find the rotated atoms")
        
        if orb_rot is None:
            orb_rot = self.get_orbital_rot_from_cartesian_rot(cart_rot)

        new_coefficients = orb_rot @ coefficients
        ordered_coeff = np.zeros_like(new_coefficients)
        coe_slice = self.atom_slices
        for current_pos, origin_pos in enumerate(from_which):
            ordered_coeff[coe_slice[current_pos]] = new_coefficients[coe_slice[origin_pos]]

        return ordered_coeff

    
    def decompose_subspace(
        self, coefficients: np.ndarray, group: ssym.SiteSymmetryGroup
    ) -> typing.Dict[str, np.ndarray]:
        maingroup = SiteSymmetryGroupwithOrbital.from_sitesymmetrygroup_irreps(
            group, self.irreps_str)
        group_order = len(maingroup.operations)
        for irrep in maingroup.irreps:
            character = irrep.characters
            transformed = []
            for v in coefficients:
                summed = np.zeros_like(v)
                for op, chi in zip(maingroup.operations, character):
                    rotated = self.symrotate_orbitalvector(
                        v, op.rot_cartesian, params.stol, op.rot_orbital
                    )
                    summed += rotated * (chi * irrep.dimension / group_order)
                transformed.append(summed)
                
            linearly_independent = tools.get_distinct_nonzero_vector_from_coefficients(
                np.vstack(transformed))
            assert len(linearly_independent) % irrep.dimension == 0  
        

def get_nonzero_independent_linear_combinations(
        inputvectors: typing.List[OrbitalsVector]) -> typing.List[OrbitalsVector]:
    """removes the linearly dependent vector in a list of input linear combinations"""
    coefficient_matrix = np.vstack([lc.coefficients for lc in inputvectors])
    distinct = tools.get_distinct_nonzero_vector_from_coefficients(coefficient_matrix)

    tpos = inputvectors[0].tpos4
    slice_ = inputvectors[0].atom_slice
    return [OrbitalsVector(tpos, slice_, row) for row in distinct]
