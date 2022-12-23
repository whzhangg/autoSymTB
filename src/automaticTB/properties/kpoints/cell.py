import numpy as np, typing
from .kmesh import Kmesh
from .kpath import Kpath, Kline


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

    def get_volume(self) -> float:
        """\
        return volume in A^3
        """
        a1 = self.cell[0]
        a2 = self.cell[1]
        a3 = self.cell[2]
        return np.cross(a1,a2).dot(a3)


    def __init__(self, cell: np.ndarray) -> None:
        self.cell = cell
        self.reciprocal_cell = self.find_RCL(cell)


    def get_Kmesh(self, nk: typing.Tuple[int, int, int]) -> Kmesh:
        return Kmesh(self.reciprocal_cell, nk)


    def recommend_kgrid(self, gridsize: int) -> typing.Tuple[int, int, int]:
        norms = np.linalg.norm(self.reciprocal_cell, axis=1)
        ratio = gridsize / np.mean(norms)
        return np.array(norms * ratio, dtype=int)


    def get_kpath_from_path_string(self, 
        kpath_string: typing.List[str], 
        quality: int = 1
    ) -> Kpath:
        """
        return a Kpath object with string in the style of wannier, eg:
        X 0.5 0.0 0.0 G 0.0 0.0 0.0
        seqarated by space only
        """
        kpaths = [Kline.from_str(s) for s in kpath_string]
        return Kpath(self.reciprocal_cell, kpaths, quality)


    def get_bandline_from_kpoint_position(self, 
        kpoint_positions: typing.Dict[str, np.ndarray],
        path_symbol: typing.List[str], 
        quality: int = 1
    ) -> Kpath:
        """
        provide kpath from a dictionary of kpoint position as well as the path symbol
        path symbol is given by ["G - X","X - K"] for example    
        """
        kpaths = []
        for apath in path_symbol:
            parts = apath.split("-")
            sym1 = parts[0].strip()
            sym2 = parts[1].strip()
            if sym1 not in kpoint_positions or sym2 not in kpoint_positions:
                raise ValueError("kpoint symbol not provided in the dictionary")
            kpaths.append(
                Kline(
                    sym1, kpoint_positions[sym1], sym2, kpoint_positions[sym2]
                )
            )
        return Kpath(self.reciprocal_cell, kpaths, quality)
