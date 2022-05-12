import abc
import numpy as np

class TightBindingBase(abc.ABC):
    @abc.abstractmethod 
    def solveE_at_k(self, k: np.ndarray) -> np.ndarray:
        # solve at a single kpoints
        pass

    def solveE_at_ks(self, ks: np.ndarray) -> np.ndarray:
        return np.vstack(
            [ self.solveE_at_k(k) for k in ks ]
        )