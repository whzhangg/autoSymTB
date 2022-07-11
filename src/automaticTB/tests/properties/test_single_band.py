import numpy as np
import typing
from automaticTB.tightbinding import Royer_Richard_TB, SingleBand_TB
from automaticTB.properties.kpath import Kline, Kpath
from automaticTB.properties.bandstructure import get_bandstructure_result, BandStructureResult
import matplotlib.pyplot as plt

def prepare_cubic_kpath() -> Kpath:
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
    return kpath


def test_plot_bandstructure_single_band():
    testfilename = "Band_singleband.pdf"
    kpath = prepare_cubic_kpath()
    overlaps = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
    Es = []
    for overlap in overlaps:
        SingleBand_TB.overlap = overlap
        tb = SingleBand_TB()
        band_result = get_bandstructure_result(tb, kpath)
        x = band_result.x
        Es.append(band_result.E)
        ticks = band_result.ticks
    
    fig = plt.figure()
    axes = fig.subplots()

    axes.set_title("Single Band Dispersion with Different Overlaps")
    axes.set_ylabel("Energy (eV)")
    ymin = np.min(Es)
    ymax = np.max(Es)
    ymin = ymin - (ymax - ymin) * 0.05
    ymax = ymax + (ymax - ymin) * 0.05

    axes.set_xlim(min(x), max(x))
    axes.set_ylim(ymin, ymax)

    for overlap, E in zip(overlaps, Es):
        axes.plot(x, E, label = str(overlap))
    for tick in ticks:
        x = tick.xpos
        axes.plot([x,x], [ymin,ymax], color='gray')

    tick_x = [ tick.xpos for tick in ticks ]
    tick_s = [ tick.symbol for tick in ticks ]
    axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
    axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))
    axes.legend()
    fig.savefig(testfilename)
    
if __name__ == "__main__":
    test_plot_bandstructure_single_band()