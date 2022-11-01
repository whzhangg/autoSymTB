from automaticTB.tightbinding.models import get_singleband_tightbinding_with_overlap
import numpy as np
import typing

overlap = 0.04

singleband_kpoint_energy: \
typing.List[typing.Tuple[np.ndarray, float]] = [
        (np.array([0.5,0.5,0.0]), 2.17391304), # M
        (np.array([0.5,0.0,0.0]),-1.85185185), # X
        (np.array([0.0,0.0,0.0]),-4.83870968), # G
        (np.array([0.5,0.5,0.5]), 7.89473684), # R
    ]


def test_solve_singleband():
    """we solve the single band dispersion at 4 k point and test the results"""
    singleband_tb = get_singleband_tightbinding_with_overlap(overlap)

    for kp, e in singleband_kpoint_energy:
        eigen_gamma, coefficient = singleband_tb.solveE_at_k(kp)
        assert np.isclose(eigen_gamma[0], e, atol=1e-3)
        
