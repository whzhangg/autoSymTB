import numpy as np
# how small we consider to be zero

ztol = 1e-4
stol = 1e-6 # Ang. for use in crystals structure related comparsion
spgtol = 1e-6

# see https://numpy.org/doc/stable/user/basics.types.html
COMPLEX_TYPE = np.cdouble
#real_coefficient_type = np.double
#COMPLEX_TYPE = np.cdouble

#use_complex_character = True
#if use_complex_character:
#    LC_coefficients_dtype = COMPLEX_TYPE
#else:
#    LC_coefficients_dtype = real_coefficient_type
