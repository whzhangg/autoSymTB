import numpy as np
from automaticTB.properties.kpath import Kline, Kpath

kpos = {
    "M": np.array([0.5,0.5,0.0]),
    "X": np.array([0.5,0.0,0.0]),
    "G": np.array([0.0,0.0,0.0]),
    "R": np.array([0.5,0.5,0.5]),
}

lines = [
    Kline("M", kpos["M"], "R", kpos["R"]),
    Kline("R", kpos["R"], "G", kpos["G"]),
    Kline("G", kpos["G"], "X", kpos["X"]),
    Kline("X", kpos["X"], "M", kpos["M"]),
    Kline("M", kpos["M"], "G", kpos["G"]),
]

cubic_kpath = Kpath(np.eye(3), lines)


