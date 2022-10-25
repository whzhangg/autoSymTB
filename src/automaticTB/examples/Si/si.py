import numpy as np
from ...structure import Structure

__all__ = ["get_Si_structure_2s2p", "get_Si_structure_2s2p3s"]

fcc_si = {
    "cell": np.eye(3) * 5.43053,
    "positions": np.array(
       [[0.25, 0.25, 0.25],
        [0.,   0.,   0.  ],
        [0.25, 0.75, 0.75],
        [0.,   0.5,  0.5 ],
        [0.75, 0.25, 0.75],
        [0.5,  0.,   0.5 ],
        [0.75, 0.75, 0.25],
        [0.5,  0.5,  0.  ]]
    ),
    "types": [14, 14, 14, 14, 14, 14, 14, 14]
}


def get_Si_structure_2s2p() -> Structure:
    fcc_si = {
        "cell": np.eye(3) * 5.43053,
        "positions": np.array(
        [[0.25, 0.25, 0.25],
            [0.,   0.,   0.  ],
            [0.25, 0.75, 0.75],
            [0.,   0.5,  0.5 ],
            [0.75, 0.25, 0.75],
            [0.5,  0.,   0.5 ],
            [0.75, 0.75, 0.25],
            [0.5,  0.5,  0.  ]]
        ),
        "types": [14, 14, 14, 14, 14, 14, 14, 14]
    }
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    return Structure.from_cpt_rcut(c,p,t, {"Si": [[2,0],[2,1]]}, 3.0)


def get_Si_structure_2s2p3s() -> Structure:
    fcc_si = {
        "cell": np.eye(3) * 5.43053,
        "positions": np.array(
        [[0.25, 0.25, 0.25],
            [0.,   0.,   0.  ],
            [0.25, 0.75, 0.75],
            [0.,   0.5,  0.5 ],
            [0.75, 0.25, 0.75],
            [0.5,  0.,   0.5 ],
            [0.75, 0.75, 0.25],
            [0.5,  0.5,  0.  ]]
        ),
        "types": [14, 14, 14, 14, 14, 14, 14, 14]
    }
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    
    return Structure.from_cpt_rcut(
        c, p, t, {"Si": [[2, 0],[2, 1], [3, 0]]}, 3.0
    )
