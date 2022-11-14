from .solve.structure import Structure
from .solve.interaction import CombinedAOSubspaceInteraction
from .solve.functions import get_EquationSystem_from_Structure
import numpy as np, typing, dataclasses
from .tools import read_cif_to_cpt

@dataclasses.dataclass
class AtomicOrbital:
    pindex: int
    n: int 
    l: int
    m: int
    translation: np.ndarray(int)

    def as_line(self) -> str:
        return "{:>3d}{:>3d}{:>3d}{:>3d} t = {:>3d}{:>3d}{:>3d}".format(
            self.p_index, self.n, self.l, self.m, *self.translation
        )

    @classmethod
    def from_line(cls, aline) -> "AtomicOrbital":
        indices, translation = aline.split("t =") 
        idx = indices.split()
        trans = np.array(translation.split(), dtype=int)
        return cls(
            int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]), 
            trans
        )


class ParameterPropertyRelationship:
    """
    this class is the interface between the solve class and the tightbinding and property class
    """
    @classmethod
    def from_cpt_orbitals_rcut(cls, 
        cell: np.ndarray, positions: np.ndarray, types: np.ndarray,
        orbitals_dict: typing.Dict[str, str],
        rcut: typing.Optional[float] = None
    ) -> "ParameterPropertyRelationship":
        structure = Structure(cell, positions, types, orbitals_dict, rcut)
        equation_system = get_EquationSystem_from_Structure(structure)
        return cls(structure, equation_system)

    @classmethod
    def from_cif_orbitals_rcut(cls, 
        cif_filename: str, 
        orbitals_dict: typing.Dict[str, str],
        rcut: typing.Optional[float] = None
    ) -> "ParameterPropertyRelationship":
        c, p, t = read_cif_to_cpt(cif_filename)
        return ParameterPropertyRelationship.from_cpt_orbitals_rcut(
            c, p, t ,orbitals_dict, rcut
        )

    def __init__(self, struct: Structure, equation: CombinedAOSubspaceInteraction):
        self.cell = struct.cell
        self.positions = struct.positions
        self.types = struct.types
        self._combinedequation = equation

    def write_data(filename):
        pass

    