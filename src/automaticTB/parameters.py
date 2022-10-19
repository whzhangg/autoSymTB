import numpy as np
import torch
# how small we consider to be zero
zero_tolerance = 1e-6

# spg
symprec = 1e-4

use_complex_character = True

real_coefficient_type = np.float64
complex_coefficient_type = np.complex128

if use_complex_character:
    LC_coefficients_dtype = complex_coefficient_type
else:
    LC_coefficients_dtype = real_coefficient_type

torch_float = torch.float
torch_int = torch.int
torch_device = "cpu"