import numpy as np
# how small we consider to be zero
zero_tolerance = 1e-6

# spg
symprec = 1e-4

use_complex_character = True

if use_complex_character:
    LC_coefficients_dtype = np.complex128
else:
    LC_coefficients_dtype = np.float64