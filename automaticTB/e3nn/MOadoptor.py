import torch
from torch_geometric.data import Data
from ..SALCs import LinearCombination
from ..atomic_orbitals import Orbitals, OrbitalsList
from ..utilities import Pair
from ..parameters import real_coefficient_type, torch_float, torch_int
import numpy as np
# in this script, we make a pair of MO into a Data 

embed_atomic_orbital_coefficients = Orbitals([0,1,2])

def get_embedded_AOs(lc: LinearCombination) -> np.ndarray:

    embedded_coefficients = np.zeros(
        (len(lc.sites),len(embed_atomic_orbital_coefficients.sh_list)),
        dtype=real_coefficient_type
    )
    
    embed_l_slice = embed_atomic_orbital_coefficients.slice_dict
    for i, subspaceslice in enumerate(lc.orbital_list.subspace_slice_dict):
        for l_given, slice_given in subspaceslice.items():
            embed_position = embed_l_slice[l_given]
            embedded_coefficients[i, embed_position] = lc.coefficients[slice_given].real
    
    return embedded_coefficients

def get_geometric_data_from_pair_linear_combination(mopair: Pair) -> Data:
    left: LinearCombination = mopair.left
    right: LinearCombination = mopair.right

    positions = np.vstack([site.pos for site in left.sites])


    atomic_number = torch.tensor([site.atomic_number for site in left.sites], dtype = torch_int)
    x_left = torch.tensor(get_embedded_AOs(left), dtype=torch_float)
    x_right = torch.tensor(get_embedded_AOs(right), dtype=torch_float)
    y = torch.tensor(0, dtype=torch_float)
    pos = torch.tensor(positions, dtype = torch_float)
    norms = torch.linalg.norm(pos, dim = 1)
    origin_index = (norms==0).nonzero().squeeze()
    return Data(
        x_left = x_left, x_right = x_right, origin_index = origin_index, pos = pos, atomic_number = atomic_number
    )




