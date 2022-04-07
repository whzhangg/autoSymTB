from typing import Union, Tuple
from e3nn.o3 import Irreps
from torch_type import float_type, int_type
import torch

class Symmetry:
    def __init__(self, euler_angle: Tuple[float], inversion: int) -> None:
        assert len(euler_angle) == 3
        self._euler_angle = torch.tensor(euler_angle, dtype = float_type)
        self._inversion = torch.tensor(inversion, dtype = int_type)

    @property
    def euler_angle(self):
        # only one element tensors can use .item()
        return self._euler_angle.tolist()

    @property
    def inversion(self):
        return self._inversion.item()

    def get_matrix(self, irreps: Union[Irreps, str]):
        rep = Irreps(irreps)
        return rep.D_from_angles(*self._euler_angle, self._inversion) 
