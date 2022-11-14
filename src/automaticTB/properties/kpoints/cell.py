import numpy as np, typing
from .kmesh import Kmesh
from .kpath import Kpath

class UnitCell:
    @staticmethod
    def find_RCL(cell: np.ndarray) -> np.ndarray:
        """find the reciprocal lattice vectors"""
        a1 = cell[0]
        a2 = cell[1]
        a3 = cell[2]
        volumn=np.cross(a1,a2).dot(a3)
        b1=(2*np.pi/volumn)*np.cross(a2,a3)
        b2=(2*np.pi/volumn)*np.cross(a3,a1)
        b3=(2*np.pi/volumn)*np.cross(a1,a2)
        return np.vstack([b1,b2,b3])


    def __init__(self, cell: np.ndarray) -> None:
        self.cell = cell
        self.reciprocal_cell = self.find_RCL(cell)


    def get_Kmesh(self, nk: typing.Tuple[int, int, int]) -> Kmesh:
        return Kmesh(self.reciprocal_cell, nk)


    def get_bandline(self, bandline_strings: typing.List[str]) -> Kpath:
        return Kpath.from_cell_string(self.cell, bandline_strings)
