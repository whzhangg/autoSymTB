from automaticTB.properties import TetraDOS, get_tetrados_result
from automaticTB.tightbinding.models import get_singleband_tightbinding_with_overlap

tb = get_singleband_tightbinding_with_overlap(0.04) # default overlap
ngrid = [20, 20, 20]

def test_singleband_dos():
    result = get_tetrados_result(tb, ngrid, x_density=60)
    result.plot_data("result_singleband_dos.pdf")

