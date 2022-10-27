import numpy as np
from automaticTB.properties.kpath import Kpath

__all__ = ["cubic_kpath"]

# Cubic path

cubic_paths_str = [ 
    "M 0.5 0.5 0.0  R 0.5 0.5 0.5",
    "R 0.5 0.5 0.5  G 0.0 0.0 0.0",
    "G 0.0 0.0 0.0  X 0.5 0.0 0.0",
    "X 0.5 0.0 0.0  M 0.5 0.5 0.0",
    "M 0.5 0.5 0.0  G 0.0 0.0 0.0"
]

cubic_kpath = Kpath.from_cell_string(
    np.eye(3), cubic_paths_str
)

'''
cubic_kpos = {
    "M": np.array([0.5,0.5,0.0]),
    "X": np.array([0.5,0.0,0.0]),
    "G": np.array([0.0,0.0,0.0]),
    "R": np.array([0.5,0.5,0.5]),
}

cubic_lines = [
    Kline("M", cubic_kpos["M"], "R", cubic_kpos["R"]),
    Kline("R", cubic_kpos["R"], "G", cubic_kpos["G"]),
    Kline("G", cubic_kpos["G"], "X", cubic_kpos["X"]),
    Kline("X", cubic_kpos["X"], "M", cubic_kpos["M"]),
    Kline("M", cubic_kpos["M"], "G", cubic_kpos["G"]),
]

cubic_kpath = Kpath(np.eye(3), cubic_lines)
'''