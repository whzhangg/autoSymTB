import numpy as np
from ...structure import Structure

# the cell parameter value come from https://www.nature.com/articles/s41598-019-50108-0#Sec4
# see supplementary material, (MAPbCl3)
# the structure is given by Page 7, Boyer Richard et al.

cubic_perovskite = {
    "cell": np.eye(3, dtype=float) * 5.6804,
    "positions": np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5]
    ]),
    "types": [ 82, 17, 17, 17 ]
}

def get_perovskite_structure() -> Structure:
    orbits_used = {
        "Pb": [[1,0],[1,1]],
        "Cl": [[1,0],[1,1]]
    }
    return Structure.from_cpt_rcut(
        cubic_perovskite["cell"], 
        cubic_perovskite["positions"], 
        cubic_perovskite["types"], orbits_used, 3.0)
