from automaticTB.tightbinding.models import Royer_Richard_TB
from automaticTB.properties import FatBandResult, Kpath

tb = Royer_Richard_TB()
paths = [ 
    "M 0.5 0.5 0.0  R 0.5 0.5 0.5",
    "R 0.5 0.5 0.5  G 0.0 0.0 0.0",
    "G 0.0 0.0 0.0  X 0.5 0.0 0.0",
    "X 0.5 0.0 0.0  M 0.5 0.5 0.0",
    "M 0.5 0.5 0.0  G 0.0 0.0 0.0"
]

def test_plot_bandstructure():
    testfilename = "FatBand_RoyerRichardPerovskite.pdf"
    fatband_dict = {
        "Pb s": ["Pb(1) s"],
        "Cl p": ["Cl(2) px","Cl(2) py","Cl(2) pz"]
    }

    kpath = Kpath.from_cell_string(tb.cell, paths)
    band_result = FatBandResult.from_tightbinding_and_kpath(tb, kpath)
    band_result.plot_fatband(testfilename, fatband_dict)
