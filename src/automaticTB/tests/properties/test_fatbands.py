import numpy as np
import typing
from automaticTB.tightbinding import Royer_Richard_TB
from automaticTB.properties.kpath import Kpath, Kline
from automaticTB.properties.bandstructure import FlatBandResult


def prepare_tightbinding_kpath() -> typing.Tuple[Royer_Richard_TB, Kpath]:
    # as in the paper of 
    # Symmetry-Based Tight Binding Modeling of Halide Perovskite Semiconductors
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

    kpath = Kpath(np.eye(3), lines)
    tb = Royer_Richard_TB()
    return tb, kpath


def test_plot_bandstructure():
    testfilename = "FatBand_RoyerRichardPerovskite.pdf"
    fatband_dict = {
        "Pb s": ["Pb(1) s"],
        "Cl p": ["Cl(2) px","Cl(2) py","Cl(2) pz"]
    }
    tb, kpath = prepare_tightbinding_kpath()
    band_result = FlatBandResult.from_tightbinding_and_kpath(tb, kpath)
    band_result.plot_fatband(testfilename, fatband_dict)
