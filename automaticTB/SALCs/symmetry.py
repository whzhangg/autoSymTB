from typing import Union, Tuple
from e3nn.o3 import Irreps
from torch_type import float_type, int_type
import torch
from atomic_orbital import CenteredAO, LinearCombination
import math

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

    def rotate_AO(self, AO: CenteredAO) -> Tuple[torch.Tensor]:
        """return the rotated AOs, as well as the coefficients for each orbital"""
        rotation_cartesian = self.get_matrix("1x1o")
        pos2 = torch.einsum("ij, j -> i", rotation_cartesian, AO.position)
        orbital_coefficient = self.get_matrix(AO.irreps).T
        return CenteredAO(AO.atom_symbol, pos2, AO.irreps), orbital_coefficient

    def rotate_LC(self, LC: LinearCombination) -> LinearCombination:
        """return the rotated linear combination"""
        new_coefficient = []
        new_AOlist = []
        for i, iao in enumerate(LC.AOs):
            new_ao, rotated_coeff = self.rotate_AO(iao)
            new_AOlist.append(new_ao)
            for iorbit in range(iao.irreps.dim):
                rotated_coeff[iorbit, :] *= LC.coefficient[LC.get_AO_index(i, iorbit)]
            new_coefficient.append(torch.sum(rotated_coeff, dim=0))
        
        sorted_coefficients = []
        for iao in LC.AOs:
            for j, jao in enumerate(new_AOlist):
                if iao == jao:
                    sorted_coefficients.append(new_coefficient.pop(j))
                    new_AOlist.pop(j)
                    continue
        sorted_coefficients = torch.hstack(sorted_coefficients)
        return LinearCombination(sorted_coefficients, LC.AOs)

if __name__ == "__main__":
    from atomic_orbital import get_coefficient_dimension, get_linear_combination, VectorSpace, CenteredAO
    D3h_positions = torch.tensor([
        [ 2.0,   0.0, 0.0] ,
        [-1.0, math.sqrt(3), 0.0],
        [-1.0,-math.sqrt(3), 0.0],
    ], dtype = float_type)
    D3h_positions = torch.index_select(D3h_positions, 1, torch.tensor([0,2,1]))
    D3h_AOs = [ '1x0e' for pos in D3h_positions ]

    aolist = []
    for i, (pos, ao) in enumerate(zip(D3h_positions, D3h_AOs)):
        aolist.append(CenteredAO(f"C{i+1}", pos, ao))
    n = get_coefficient_dimension(aolist)
    lc = get_linear_combination(torch.eye(n, dtype = float_type), aolist)

    AO_vector_space = VectorSpace(lc)
    #print(AO_vector_space.display_str)

    Possible_symmetry = [
        Symmetry((0,       0,            0), 0), #0 E        identity
        Symmetry((0,       0,   torch.pi/3), 1), #3 S3_1     
        Symmetry((0,       0, 2*torch.pi/3), 0), #2 C3_1       120degree rotation
        Symmetry((0,       0,     torch.pi), 1), #1 Sigma_h  horizontal refection
        Symmetry((0,       0, 4*torch.pi/3), 0), #2 C3       120degree rotation    
        Symmetry((0,       0, 5*torch.pi/3), 1), #3 S3_2    
        Symmetry((            0, torch.pi,             0), 0), #4 C2'_1      rotation around x 
        Symmetry(( 2*torch.pi/3, torch.pi, -2*torch.pi/3), 0), #4 C2'_2       
        Symmetry((-2*torch.pi/3, torch.pi,  2*torch.pi/3), 0), #4 C2'_3      
        Symmetry((            0, torch.pi, -torch.pi), 1),              #5 Sigma_v  vertical reflection
        Symmetry((-2*torch.pi/3, torch.pi, -torch.pi), 1),  #5 Sigma_v  vertical reflection
        Symmetry(( 2*torch.pi/3, torch.pi, torch.pi), 1),   #5 Sigma_v  vertical reflection
    ]

    aos = lc[0]

    for sym in Possible_symmetry:
        print(sym.rotate_LC(aos))