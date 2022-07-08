import numpy as np
import typing
from automaticTB.tightbinding import Royer_Richard_TB
from automaticTB.properties.bandstructure import Kpath, Kline, get_bandstructure_result, BandStructureResult
from automaticTB.properties.fatband import get_fatband_result


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
        Kline("M", kpos["M"], "R", kpos["R"], 10),
        Kline("R", kpos["R"], "G", kpos["G"], 18),
        Kline("G", kpos["G"], "X", kpos["X"], 10),
        Kline("X", kpos["X"], "M", kpos["M"], 10),
        Kline("M", kpos["M"], "G", kpos["G"], 15),
    ]

    kpath = Kpath(np.eye(3), lines)
    tb = Royer_Richard_TB()
    return tb, kpath


def test_plot_bandstructure():
    testfilename = "FatBand_RoyerRichardPerovskite.pdf"
    tb, kpath = prepare_tightbinding_kpath()
    band_result = get_fatband_result(tb, kpath)
    band_result.plot_data(testfilename, [["Pb(1) s"],["Cl(2) px","Cl(2) py","Cl(2) pz"]])
