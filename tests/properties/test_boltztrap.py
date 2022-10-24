import numpy as np
from automaticTB.properties.boltztrap import TBoltzTrapCalculation, write_results_yaml
from automaticTB.tightbinding.models import get_singleband_tightbinding_with_overlap


def test_single_band_dos():
    tb = get_singleband_tightbinding_with_overlap(0.04)
    boltztrap = TBoltzTrapCalculation(
        tb, 
        [10, 10, 10],
        nele = 1,
        dosweight = 2
    )
    result = boltztrap.calculate(
        [300], [1.3, 1.4, 1.5], 1.0, 1.0
    )
    write_results_yaml(result)
    print(result)


if __name__ == "__main__":
    test_single_band_dos()