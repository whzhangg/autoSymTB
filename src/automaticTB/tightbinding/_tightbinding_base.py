import typing, abc
import numpy as np
from .hij import Pindex_lm
from ._secular_solver import solve_secular_sorted

class TightBindingBase(abc.ABC):

    @property
    @abc.abstractmethod
    def cell(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def positions(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def types(self) -> typing.List[int]:
        pass

    @property
    @abc.abstractmethod
    def basis(self) -> typing.List[Pindex_lm]:
        pass

    @abc.abstractmethod 
    def Hijk(self, frac_k: np.ndarray) -> np.ndarray:
        pass 

    @abc.abstractmethod
    def Sijk(self, frac_k: np.ndarray) -> np.ndarray:
        pass

    def solveE_at_k(self, k: np.ndarray) -> np.ndarray:
        # solve at a single kpoints
        h = self.Hijk(k)
        #w, v = np.linalg.eig(h)
        s = self.Sijk(k)
        w, c = solve_secular_sorted(h, s)
        return w.real, c

    def solveE_c2_at_ks(self, ks: np.ndarray) -> np.ndarray:
        ek = []
        coeff = []
        for k in ks:
            w, complex_c =self.solveE_at_k(k)
            ek.append(w)
            c_T = (np.abs(complex_c)**2).T # [ibnd, iorb]
            coeff.append(c_T[np.newaxis,...])
        ek = np.vstack(ek)
        coeff = np.vstack(coeff)
        assert coeff.shape[0] == ek.shape[0]
        return ek, coeff

    def solveE_at_ks(self, ks: np.ndarray) -> np.ndarray:
        return np.vstack(
            [ self.solveE_at_k(k)[0] for k in ks ]
        )
