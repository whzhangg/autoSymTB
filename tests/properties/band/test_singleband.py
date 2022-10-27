from automaticTB.tightbinding.models import get_singleband_tightbinding_with_overlap
from automaticTB.functions import wrtie_bandstructure_from_tightbinding

def test_band():
    overlap = 0.04
    tb = get_singleband_tightbinding_with_overlap(overlap)
    paths = [ 
        "M 0.5 0.5 0.0  R 0.5 0.5 0.5",
        "R 0.5 0.5 0.5  G 0.0 0.0 0.0",
        "G 0.0 0.0 0.0  X 0.5 0.0 0.0",
        "X 0.5 0.0 0.0  M 0.5 0.5 0.0",
        "M 0.5 0.5 0.0  G 0.0 0.0 0.0"
    ]

    wrtie_bandstructure_from_tightbinding(
        tb, "singleband", paths,
        emin = None, 
        emax = None, 
        quality = 0,
        make_folder = False 
    )
