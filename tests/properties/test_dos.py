from automaticTB.properties import TetraDOS, get_tetrados_result
from automaticTB.tightbinding.models import SingleBand_TB

tb = SingleBand_TB() # default overlap
ngrid = [20, 20, 20]

def test_singleband_dos():
    result = get_tetrados_result(tb, ngrid, x_density=60)
    result.plot_data("result_singleband_dos.pdf")

