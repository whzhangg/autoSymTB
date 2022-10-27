from automaticTB.functions import write_DOS_from_tightbinding
from automaticTB.tightbinding.models import get_singleband_tightbinding_with_overlap

def test_dos():
    tb = get_singleband_tightbinding_with_overlap(0.04) # default overlap
    write_DOS_from_tightbinding(
        tb, 
        "singleband", 
        emin = None, 
        emax = None,
        gridsize = 30,
        xdensity = 50,
        make_folder=False
    )

