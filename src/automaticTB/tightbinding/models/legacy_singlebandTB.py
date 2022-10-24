# a single band tightbinding with energy at zero

import numpy as np
import typing
from ..tightbinding_model import TightBindingBase, Pindex_lm

a = 0.0
b = -1.0

HijR = {
    (0, 0, 0): {
        ("s","s"): a
    },
    (1, 0, 0): {
        ("s","s"): b
    },
    (-1, 0, 0): {
        ("s","s"): b
    },
    (0, 1, 0): {
        ("s","s"): b
    },
    (0, -1, 0): {
        ("s","s"): b
    },
    (0, 0, 1): {
        ("s","s"): b
    },
    (0, 0, -1): {
        ("s","s"): b
    },
}

class SingleBand_TB(TightBindingBase):
    """
    it return the model given in the article
    """
    orbits = ["s"]
    orbit_indices = {o:io for io, o in enumerate(orbits)}
    overlap = 0.04

    def __init__(self) -> None:
        self._cell = np.eye(3)
        self._position = np.array([[0.5,0.5,0.5]])
        self.size = len(self.orbits)

    @property
    def positions(self) -> np.ndarray:
        return self._position

    @property
    def cell(self) -> np.ndarray:
        return self._cell

    @property
    def types(self) -> typing.List[int]:
        return [1]  

    @property
    def basis(self) -> typing.List[Pindex_lm]:
        return [
            Pindex_lm(0, 0, 0, np.array([0,0,0]))
        ]

    def Hijk(self, k: typing.Tuple[float]):
        k = np.array(k)
        ham = np.zeros((self.size, self.size), dtype=complex)
        for R, Hij in HijR.items():
            r = np.array(R)
            for (iorb, jorb), value in Hij.items():
                i = self.orbit_indices[iorb]
                j = self.orbit_indices[jorb]
                tau_i = self._position[(i+1)//4 - 1]
                tau_j = self._position[(j+1)//4 - 1]

                kR = np.dot(k, r + tau_j - tau_i)
                ham[i,j] += value * np.exp(2j * np.pi * kR)
        return ham

    def Sijk(self, k: typing.Tuple[float]):
        """
        set SingleBand_TB.overlap to adjust the overlap strength, it 
        should be approximately in range [0, 0.1] and positive
        """
        SijR = {
            (0, 0, 0): {
                ("s","s"): 1.0
            },
            (1, 0, 0): {
                ("s","s"): self.overlap
            },
            (-1, 0, 0): {
                ("s","s"): self.overlap
            },
            (0, 1, 0): {
                ("s","s"): self.overlap
            },
            (0, -1, 0): {
                ("s","s"): self.overlap
            },
            (0, 0, 1): {
                ("s","s"): self.overlap
            },
            (0, 0, -1): {
                ("s","s"): self.overlap
            },
        }
        k = np.array(k)
        overlap = np.zeros((self.size, self.size), dtype=complex)
        for R, Sij in SijR.items():
            r = np.array(R)
            for (iorb, jorb), value in Sij.items():
                i = self.orbit_indices[iorb]
                j = self.orbit_indices[jorb]
                tau_i = self._position[(i+1)//4 - 1]
                tau_j = self._position[(j+1)//4 - 1]

                kR = np.dot(k, r + tau_j - tau_i)
                overlap[i,j] += value * np.exp(2j * np.pi * kR)
        return overlap