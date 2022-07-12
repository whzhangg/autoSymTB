import numpy as np
import typing
from automaticTB.tightbinding.models import Royer_Richard_TB
from automaticTB.properties.bandstructure import FatBandResult
from prepare_data import cubic_kpath as kpath

tb = Royer_Richard_TB()

def test_plot_bandstructure():
    testfilename = "result_FatBand_RoyerRichardPerovskite.pdf"
    fatband_dict = {
        "Pb s": ["Pb(1) s"],
        "Cl p": ["Cl(2) px","Cl(2) py","Cl(2) pz"]
    }
    band_result = FatBandResult.from_tightbinding_and_kpath(tb, kpath)
    band_result.plot_fatband(testfilename, fatband_dict)
