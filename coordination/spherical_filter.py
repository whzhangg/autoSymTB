from torch import float32
from e3nn.o3 import Irreps, spherical_harmonics, rand_matrix
import torch

vector_tet = torch.tensor([[ 1, 1, 1],
          [-1,-1, 1],
          [ 1,-1,-1],
          [-1, 1,-1]], dtype = float)

vector_oct = torch.tensor(
    [
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1],
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0,-1]
    ], dtype = float32
)

vector_oct = torch.einsum('ij,zj->zi', rand_matrix().float(), vector_oct)

sh = Irreps.spherical_harmonics(5)
harmonic = spherical_harmonics(
            l=sh,
            x=vector_oct,
            normalize=True,  # here we don't normalize otherwise it would not be a polynomial
            normalization='component'
        )

print(harmonic)
print(torch.sum(harmonic, dim = 0))