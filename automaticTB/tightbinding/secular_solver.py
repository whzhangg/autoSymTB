import numpy as np
import typing
import scipy.linalg as scipylinalg
# given a secular equation $H_{\mu\nu}(k)$ and $S_{\mu\nu}(k)$, we attempt to solve for the coefficients C
r"""
we require that the result to be similar to this form w, v = np.linalg.eig(ham)
Method: Ideas of Quantum Chemistry Appendix L.
(H - eS) c = 0
(S^{-1/2} H - e S^{-1/2} S) c = 0
(S^{-1/2} H S^{-1/2}S^{1/2} - e S^{1/2}) c = 0
(S^{-1/2} H S^{-1/2} - e) S^{1/2} c = 0
(\tilde{H} - e) (S^{1/2} c) = 0
"""

def solve_secular(H: np.ndarray, S: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    if H.shape != S.shape: raise "Shape must the same"
    S_minus_half = scipylinalg.fractional_matrix_power(S, -0.5)
    
    tilde_H = S_minus_half @ H @ S_minus_half
    w, v = np.linalg.eig(tilde_H)
    c = np.einsum("ij, jk -> ik", S_minus_half, v)

    return w, c

def test_secular_solver():
    H = np.array([
        [3, -5, 0],
        [-5, 3,-5],
        [0, -5, 3]
    ]) + np.random.random((3,3))

    S = np.eye(3) + np.random.random((3,3))

    w, c = solve_secular(H, S)

    Hc = np.einsum("ij, jk -> ik", H, c)
    Sc = np.einsum("ij, jk -> ik", S, c)
    eSc = w * Sc
    assert np.allclose(np.zeros_like(Hc), np.abs(Hc - eSc))

test_secular_solver()