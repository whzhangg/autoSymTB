import numpy as np
import torch
# how small we consider to be zero

zero_tolerance = 1e-4
precision_decimal = 10
# spg
tolerance_structure = 1e-4 # Ang. for use in crystals structure related comparsion


# see https://numpy.org/doc/stable/user/basics.types.html
real_coefficient_type = np.double
complex_coefficient_type = np.cdouble

use_complex_character = True
if use_complex_character:
    LC_coefficients_dtype = complex_coefficient_type
else:
    LC_coefficients_dtype = real_coefficient_type

torch_float = torch.float
torch_int = torch.int
torch_device = "cpu"