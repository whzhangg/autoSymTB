# basically, this script will dress the site symmetry group so that we do not need to calculate wigner matrix again and again.
import dataclasses, typing
from ..sitesymmetry import SiteSymmetryGroup, subduction_data
from ..rotation import orbital_rotation_from_symmetry_matrix
import numpy as np

@dataclasses.dataclass
class CartesianOrbitalRotation:
    irrep_str: str
    rot_cartesian: np.ndarray
    rot_orbital: np.ndarray    # this will be a rotation of the whole orbitals


@dataclasses.dataclass
class IrreducibleRep:
    name: str
    dimension: int
    characters: typing.List[typing.Union[float, complex]]


@dataclasses.dataclass
class SiteSymmetryGroupwithOrbital:
    groupname: str
    irreps: typing.List[IrreducibleRep]
    operations: typing.List[CartesianOrbitalRotation]
    irrep_str: str
    #
    sitesymmetrygroup: SiteSymmetryGroup 

    @classmethod
    def from_sitesymmetrygroup_irreps(cls, sitesymmetrygroup: SiteSymmetryGroup, irrep_str: str) \
    -> "SiteSymmetryGroupwithOrbital":
        irreducible_reps = []
        for irrep_name in sitesymmetrygroup.irreps.keys():
            irreducible_reps.append(
                IrreducibleRep(irrep_name, 
                    sitesymmetrygroup.irrep_dimension[irrep_name], 
                    sitesymmetrygroup.irreps[irrep_name]
                )
            )
        cartesian_and_orbital = []
        for op in sitesymmetrygroup.operations:
            orbital_rotation = orbital_rotation_from_symmetry_matrix(op, irrep_str)
            cartesian_and_orbital.append(
                CartesianOrbitalRotation(irrep_str, op, orbital_rotation)
            )
        return cls(
            groupname = sitesymmetrygroup.groupname,
            irreps = irreducible_reps,
            operations = cartesian_and_orbital,
            irrep_str = irrep_str,
            sitesymmetrygroup = sitesymmetrygroup
        )
            
 
    def get_subduction(self) \
    -> "SiteSymmetryGroupwithOrbital":
        subduction = subduction_data[self.groupname]
        subgroup = self.sitesymmetrygroup.get_subgroup_from_subgroup_and_seitz_map(
                        subduction["sub"], subduction["seitz_map"]
                    )
        return SiteSymmetryGroupwithOrbital.from_sitesymmetrygroup_irreps(subgroup, self.irrep_str)
