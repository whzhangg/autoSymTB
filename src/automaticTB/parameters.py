import numpy as np
# how small we consider to be zero

ztol = 1e-6 # smaller value fails, perhaps due to the error in rotation matrix? 
stol = 1e-6 # Ang. for use in crystals structure related comparsion
spgtol = 1e-6

# see https://numpy.org/doc/stable/user/basics.types.html
COMPLEX_TYPE = np.cdouble
