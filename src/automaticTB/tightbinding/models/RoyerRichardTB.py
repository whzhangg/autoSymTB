import numpy as np
import typing
from .._tightbinding_base import TightBindingBase
from ..hij import Pindex_lm


__all__ = ["Royer_Richard_TB"]


parameters = {
        "V_ss"   :  -1.10,
        "V_s0p1" :   1.10,
        "V_p0s1" :   0.70,
        "V_pps"  :  -3.65,
        "V_ppp"  :   0.55,
        "E_s0"   :  -9.01,
        "E_s1"   : -13.01,
        "E_p0"   :   2.34,
        "E_p1"   :  -1.96
    }


HijR = {
    ( 0, 0, 0): {
        ("S0","S0"): parameters["E_s0"],
        ("X0","X0"): parameters["E_p0"],
        ("Y0","Y0"): parameters["E_p0"],
        ("Z0","Z0"): parameters["E_p0"],
        ("S1","S1"): parameters["E_s1"],
        ("X1","X1"): parameters["E_p1"],
        ("Y1","Y1"): parameters["E_p1"],
        ("Z1","Z1"): parameters["E_p1"],
        ("S2","S2"): parameters["E_s1"],
        ("X2","X2"): parameters["E_p1"],
        ("Y2","Y2"): parameters["E_p1"],
        ("Z2","Z2"): parameters["E_p1"],
        ("S3","S3"): parameters["E_s1"],
        ("X3","X3"): parameters["E_p1"],
        ("Y3","Y3"): parameters["E_p1"],
        ("Z3","Z3"): parameters["E_p1"],
        ("S0","S1"): parameters["V_ss"],
        ("S0","S2"): parameters["V_ss"],
        ("S0","S3"): parameters["V_ss"],
        ("S0","X1"): parameters["V_s0p1"],
        ("S0","Y2"): parameters["V_s0p1"],
        ("S0","Z3"): parameters["V_s0p1"],
        ("X0","S1"): parameters["V_p0s1"],
        ("Y0","S2"): parameters["V_p0s1"],
        ("Z0","S3"): parameters["V_p0s1"],
        ("X0","X1"): parameters["V_pps"],
        ("Y0","Y2"): parameters["V_pps"],
        ("Z0","Z3"): parameters["V_pps"],
        ("X0","X2"): parameters["V_ppp"],
        ("X0","X3"): parameters["V_ppp"],
        ("Y0","Y1"): parameters["V_ppp"],
        ("Y0","Y3"): parameters["V_ppp"],
        ("Z0","Z1"): parameters["V_ppp"],
        ("Z0","Z2"): parameters["V_ppp"],
        ("S1","S0"): parameters["V_ss"],
        ("S2","S0"): parameters["V_ss"],
        ("S3","S0"): parameters["V_ss"],
        ("X1","S0"): parameters["V_s0p1"],
        ("Y2","S0"): parameters["V_s0p1"],
        ("Z3","S0"): parameters["V_s0p1"],
        ("S1","X0"): parameters["V_p0s1"],
        ("S2","Y0"): parameters["V_p0s1"],
        ("S3","Z0"): parameters["V_p0s1"],
        ("X1","X0"): parameters["V_pps"],
        ("Y2","Y0"): parameters["V_pps"],
        ("Z3","Z0"): parameters["V_pps"],
        ("X2","X0"): parameters["V_ppp"],
        ("X3","X0"): parameters["V_ppp"],
        ("Y1","Y0"): parameters["V_ppp"],
        ("Y3","Y0"): parameters["V_ppp"],
        ("Z1","Z0"): parameters["V_ppp"],
        ("Z2","Z0"): parameters["V_ppp"],
    },
    (-1, 0, 0): {
        ("S0","S1"): parameters["V_ss"],
        ("S0","X1"):-parameters["V_s0p1"],
        ("X0","S1"):-parameters["V_p0s1"],
        ("X0","X1"): parameters["V_pps"],
        ("Y0","Y1"): parameters["V_ppp"],
        ("Z0","Z1"): parameters["V_ppp"]
    },
    ( 1, 0, 0): {
        ("S1","S0"): parameters["V_ss"],
        ("X1","S0"):-parameters["V_s0p1"],
        ("S1","X0"):-parameters["V_p0s1"],
        ("X1","X0"): parameters["V_pps"],
        ("Y1","Y0"): parameters["V_ppp"],
        ("Z1","Z0"): parameters["V_ppp"]
    },
    ( 0,-1, 0): {
        ("S0","S2"): parameters["V_ss"],
        ("S0","Y2"):-parameters["V_s0p1"],
        ("Y0","S2"):-parameters["V_p0s1"],
        ("X0","X2"): parameters["V_ppp"],
        ("Y0","Y2"): parameters["V_pps"],
        ("Z0","Z2"): parameters["V_ppp"]
    },
    ( 0, 1, 0): {
        ("S2","S0"): parameters["V_ss"],
        ("Y2","S0"):-parameters["V_s0p1"],
        ("S2","Y0"):-parameters["V_p0s1"],
        ("X2","X0"): parameters["V_ppp"],
        ("Y2","Y0"): parameters["V_pps"],
        ("Z2","Z0"): parameters["V_ppp"]
    },
    ( 0, 0,-1): {
        ("S0","S3"): parameters["V_ss"],
        ("S0","Z3"):-parameters["V_s0p1"],
        ("Z0","S3"):-parameters["V_p0s1"],
        ("X0","X3"): parameters["V_ppp"],
        ("Y0","Y3"): parameters["V_ppp"],
        ("Z0","Z3"): parameters["V_pps"]
    },
    ( 0, 0, 1): {
        ("S3","S0"): parameters["V_ss"],
        ("Z3","S0"):-parameters["V_s0p1"],
        ("S3","Z0"):-parameters["V_p0s1"],
        ("X3","X0"): parameters["V_ppp"],
        ("Y3","Y0"): parameters["V_ppp"],
        ("Z3","Z0"): parameters["V_pps"]
    },
}


class Royer_Richard_TB(TightBindingBase):
    """
    it return the model given in the article
    """
    orbits = "S0 X0 Y0 Z0 S1 X1 Y1 Z1 S2 X2 Y2 Z2 S3 X3 Y3 Z3".split()
    orbit_indices = {o:io for io, o in enumerate(orbits)}

    def __init__(self) -> None:
        self._cell = np.eye(3)
        self._position = np.array([[0,0,0], [0.5,0,0], [0,0.5,0], [0, 0, 0.5]])
        self.size = len(self.orbits)

    @property
    def positions(self) -> np.ndarray:
        return self._position

    @property
    def cell(self) -> np.ndarray:
        return self._cell

    @property
    def types(self) -> typing.List[int]:
        return [82, 17, 17, 17]

    @property
    def basis(self) -> typing.List[Pindex_lm]:
        return [
            Pindex_lm(0, 2, 0, 0, np.array([0,0,0])),
            Pindex_lm(0, 2, 1,-1, np.array([0,0,0])),
            Pindex_lm(0, 2, 1, 0, np.array([0,0,0])),
            Pindex_lm(0, 2, 1, 1, np.array([0,0,0])),
            Pindex_lm(1, 2, 0, 0, np.array([0,0,0])),
            Pindex_lm(1, 2, 1,-1, np.array([0,0,0])),
            Pindex_lm(1, 2, 1, 0, np.array([0,0,0])),
            Pindex_lm(1, 2, 1, 1, np.array([0,0,0])),
            Pindex_lm(2, 2, 0, 0, np.array([0,0,0])),
            Pindex_lm(2, 2, 1,-1, np.array([0,0,0])),
            Pindex_lm(2, 2, 1, 0, np.array([0,0,0])),
            Pindex_lm(2, 2, 1, 1, np.array([0,0,0])),
            Pindex_lm(3, 2, 0, 0, np.array([0,0,0])),
            Pindex_lm(3, 2, 1,-1, np.array([0,0,0])),
            Pindex_lm(3, 2, 1, 0, np.array([0,0,0])),
            Pindex_lm(3, 2, 1, 1, np.array([0,0,0])),
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

    def Sijk(self, k: any):
        return np.eye(self.size, self.size)
