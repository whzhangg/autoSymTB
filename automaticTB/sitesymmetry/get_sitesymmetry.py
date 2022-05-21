from spglib import get_symmetry_dataset
import numpy as np
import typing
from pymatgen.core.structure import Structure
import dataclasses
from .utilities import rotation_fraction_to_cartesian

@dataclasses.dataclass
class Site:
    type: int
    pos: np.ndarray

    def __eq__(self, other) -> bool:
        return self.type == other.type and np.allclose(self.pos, other.pos)

    def rotate(self, matrix:np.ndarray):
        newpos = matrix.dot(self.pos)
        return Site(self.type, newpos)
   

def get_site_symmetry_operations(
    cell: tuple, pointgroup_operations: typing.List[np.ndarray], rcut: float = 4.0
) -> typing.List[typing.List[int]]:
    c, p, t = cell
    structure = Structure(lattice=c, coords=p, species=t)
    cartesian_pos = np.einsum("ji, kj -> ki", c, p)

    site_symmetry_operations = []
    for pos in cartesian_pos:
        # see structure.py 2791
        # and local_env.py
        nnn = structure.get_sites_in_sphere(pos, rcut)  
        nnsites = [ Site(t[n.index], n.coords - pos) for n in nnn] 
        operations = []
        for isym, sym in enumerate(pointgroup_operations):
            is_symmetry = True
            for site in nnsites:
                newsite = site.rotate(sym)
                if not newsite in nnsites: 
                    is_symmetry = False
                    break
            if is_symmetry:
                operations.append(isym)
        site_symmetry_operations.append(operations)
    
    return site_symmetry_operations


def get_site_symmetries(
    input_cpt: tuple, verification_cutoff: float = 4.0, prec = 1e-4
) -> typing.Tuple[tuple, typing.List[np.ndarray]]:
    dataset = get_symmetry_dataset(input_cpt, symprec = prec)
    cell = dataset["std_lattice"]
    positions = dataset["std_positions"]
    types = dataset["std_types"]
    fraction_rotations = np.unique(dataset["rotations"], axis=0)
    cartesian_rotations = rotation_fraction_to_cartesian(fraction_rotations, cell)

    site_symmetry_index = get_site_symmetry_operations(
        (cell, positions, types), cartesian_rotations, rcut=verification_cutoff
    )

    operations = [ cartesian_rotations[index] for index in site_symmetry_index ]
    return (cell, positions, types), operations


