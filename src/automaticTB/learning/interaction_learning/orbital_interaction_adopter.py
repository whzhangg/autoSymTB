"""
This script provide an interface to the interaction learning model, here we provide the 
orbital interaction 
"""

import typing, dataclasses
import torch
import numpy as np
from torch_geometric.data import Data
from e3nn.o3 import Irreps
from ..structure import Structure, NearestNeighborCluster
from ..parameters import torch_float, torch_device
from ..tools import Pair
from ..interaction import AO

def _get_onehot_element_encoding(
    atomic_number_in: int, atomic_number_set: typing.Set[int]
) -> np.ndarray:
    """the sequence in one hot encoding is simply given by atomic number"""
    sorted_atomic_number = sorted(list(atomic_number_set))
    encoding = np.zeros(len(sorted_atomic_number))
    encoding[sorted_atomic_number.index(atomic_number_in)] = 1
    return encoding


@dataclasses.dataclass
class ElementOneHotEncoder:
    atomic_number_set: typing.Set[int]

    def get_onehot_element_encoding(self, atomic_number_in: int) -> np.ndarray:
        """the sequence in one hot encoding is simply given by atomic number"""
        sorted_atomic_number = sorted(list(self.atomic_number_set))
        encoding = np.zeros(len(sorted_atomic_number))
        encoding[sorted_atomic_number.index(atomic_number_in)] = 1
        return encoding



class OrbitalIrrepsEncoder:

    _aoirrep_ref = {0 : "1x0e",  1 : "1x1o", 2 : "1x2e"}
    _aolists = {
        # this is used to generate a list of nSpHars from a list of NLs
        0 : [(0, 0)], 
        1 : [(1,-1), (1, 0), (1, 1)],
        2 : [(2,-2), (2,-1), (2, 0), (2, 1), (2, 2)]
    }
    def __init__(self, nl_list: typing.List[typing.Tuple[int,int]]) -> None:
        self._reps_str = ""
        self._nlm_list = []
        self._nlm_index_mapper = {}

        nl_list = list(set(nl_list))
        sorted_n = sorted(list({ n for n,l in nl_list }))
        reps = []
        for ni in sorted_n:
            sorted_l = sorted(list({l for n,l in nl_list if n == ni}))
            reps += [self._aoirrep_ref[l] for l in sorted_l]
            for li in sorted_l:
                aolist = self._aolists[li]
                self._nlm_list += [ (ni,l,m) for l,m in aolist]
        
        self._reps_str = " + ".join(reps)
        self._nlm_index_mapper = { nlm:i for i, nlm in enumerate(self._nlm_list) }

    @property
    def irreps_str(self) -> str:
        return self._reps_str


    def get_nlm_feature(self, nlm: typing.Tuple[int, int, int]) -> np.ndarray:
        n, l, m = nlm
        result = np.zeros(len(self._nlm_list))
        index = self._nlm_index_mapper[(n,l,m)]
        result[index] = 1.0
        return result



@dataclasses.dataclass
class Orbital:
    pos: np.ndarray
    element_onehot: np.ndarray
    node_feature: np.ndarray
    feature_irreps: str

    @property
    def pos_yzx(self) -> np.ndarray:
        shuffle = np.array([1,2,0]) # y z x
        return self.pos[shuffle]



@dataclasses.dataclass
class InteractingOrbitals:
    orb1: Orbital
    orb2: Orbital
    value: typing.Optional[float]

    @property
    def geometry_data(self) -> Data:
        x = np.vstack([self.orb1.node_feature, self.orb2.node_feature])
        z = np.vstack([self.orb1.element_onehot, self.orb2.element_onehot])
        pos = np.vstack([self.orb1.pos_yzx, self.orb2.pos_yzx])

        if self.value:
            y = torch.tensor(self.value).to(device = torch_device, dtype=torch_float)
        else:
            y = None

        return Data(
            x = torch.from_numpy(x).to(device = torch_device, dtype=torch_float),
            z = torch.from_numpy(z).to(device = torch_device, dtype=torch_float),
            pos = torch.from_numpy(pos).to(device = torch_device, dtype=torch_float),
            y = y
        )

    def get_randomlyrotated_pyg_data(self) -> Data:
        """hard coded random rotation"""
        
        rot_angle = torch.from_numpy(
            np.random.rand(3,1)
        ).to(device = torch_device, dtype=torch_float)

        gdata = self.geometry_data
        rotation_r = Irreps('1o').D_from_angles(*rot_angle)
        rotation_o = Irreps(self.orb1.feature_irreps).D_from_angles(*rot_angle)

        rotated_pos = torch.einsum('ij,zj->zi', rotation_r[0].float(), gdata.pos)
        rotated_orbitals = torch.einsum('ij,zj->zi', rotation_o[0].float(), gdata.x)

        return Data(
            x = rotated_orbitals,
            z = gdata.z,
            pos = rotated_pos,
            y = gdata.y
        )


def get_elementEncoder_orbitalEncoder_from_structure(structure: Structure) \
-> typing.Tuple[ElementOneHotEncoder, OrbitalIrrepsEncoder]:
    all_elements = set(structure.types)
    nl_list = []
    for nncluster in structure.nnclusters:
        for orb in nncluster.orbitalslist.orbital_list:
            nl_list += orb.nl_list
    return ElementOneHotEncoder(all_elements), OrbitalIrrepsEncoder(set(nl_list))
    

def get_interaction_orbitals_from_structure_and_AOpair_with_value(
    structure: Structure,
    AOpair: Pair,
    value: float
) -> InteractingOrbitals:
    interaction = get_interaction_orbitals_from_structure_and_AOpair(structure, AOpair)
    interaction.value = value
    return interaction

def get_interaction_orbitals_from_structure_and_AOpair(
    structure: Structure,
    AOpair: Pair
) -> InteractingOrbitals:
    ele_encoder, orb_encoder = \
        get_elementEncoder_orbitalEncoder_from_structure(structure)

    c = structure.cell
    cpos = np.einsum("ji,kj -> ki", c, structure.positions)
    
    left: AO = AOpair.left
    l_pos = cpos[left.primitive_index] + c.T.dot(left.translation)
    l_onehot = ele_encoder.get_onehot_element_encoding(left.atomic_number)
    l_feature = orb_encoder.get_nlm_feature((left.n, left.l, left.m))
    
    right: AO = AOpair.right
    r_pos = cpos[right.primitive_index] + c.T.dot(right.translation)
    r_onehot = ele_encoder.get_onehot_element_encoding(right.atomic_number)
    r_feature = orb_encoder.get_nlm_feature((right.n, right.l, right.m))

    return InteractingOrbitals(
        Orbital(l_pos, l_onehot, l_feature, orb_encoder.irreps_str),
        Orbital(r_pos, r_onehot, r_feature, orb_encoder.irreps_str),
        None
    )
