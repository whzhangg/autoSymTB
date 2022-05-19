import numpy as np

# the cell parameter value come from https://www.nature.com/articles/s41598-019-50108-0#Sec4
# see supplementary material, (MAPbCl3)
# the structure is given by Page 7, Boyer Richard et al.

cell = np.eye(3, dtype=float) * 5.6804

types = [ 82, 17, 17, 17 ]

positions = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5]
])

orbits_used = {
    "Pb": [0, 1],
    "Cl": [0, 1]
}