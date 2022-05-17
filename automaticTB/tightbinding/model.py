import abc, typing
import numpy as np
from .tightbinding_data import HijR, Pindex_lm, TBData

class TightBindingBase(abc.ABC):
    @abc.abstractmethod 
    def Hijk(self, frac_k: np.ndarray) -> np.ndarray:
        pass 

    def solveE_at_k(self, k: np.ndarray) -> np.ndarray:
        # solve at a single kpoints
        ham = self.Hijk(k)
        w, v = np.linalg.eig(ham)
        return np.sort(w.real)

    def solveE_at_ks(self, ks: np.ndarray) -> np.ndarray:
        return np.vstack(
            [ self.solveE_at_k(k) for k in ks ]
        )

class TightBindingModel(TightBindingBase):
    def __init__(self, tbd: TBData) -> None:
        self.cell = tbd.cell
        self.positions = tbd.positions
        self.types = tbd.types
        self.HijR = tbd.HijR_list

        self.basis = []
        for hijR in self.HijR:
            if hijR.left not in self.basis:
                self.basis.append(hijR.left)
                
        self.index_ref: typing.Dict[tuple, int] = {
            (basis.pindex, basis.l, basis.m):i for i, basis in enumerate(self.basis)
        }
        
    def Hijk(self, k: typing.Tuple[float]):
        k = np.array(k)
        nbasis = len(self.basis)
        ham = np.zeros((nbasis,nbasis), dtype=complex)
        for HijR in self.HijR:
            r = HijR.right.translation
            index_i = self.index_ref[
                (HijR.left.pindex, HijR.left.l, HijR.left.m)
            ]
            index_j = self.index_ref[
                (HijR.right.pindex, HijR.right.l, HijR.right.m)
            ]
            kR = np.dot(k, r + self.positions[HijR.right.pindex] - self.positions[HijR.left.pindex])
            ham[index_i,index_j] += HijR.value * np.exp(2j * np.pi * kR)
        return ham
