import os
from platform import node
import numpy as np
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, FullTensorProduct
from e3nn.nn import Gate, FullyConnectedNet, NormActivation
from e3nn.math import soft_one_hot_linspace

# this script test the ability of e3nn to find a vector of an arrow (0 -> 1)
# the input shape is basically:
#     ^ x
#     *
#  *     *   -> y
#
#     *
# the target is a vector pointing towards the longer edge

point = torch.tensor([[ 1.0, 0.0, 0.0],
                      [ 0.0, 0.1, 0.0],
                      [ 0.0,-0.1, 0.0],
                      [-2.0, 0.0, 0.0]])
node_feature = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
result = torch.tensor([-1.0, 0.0, 0.0])

rot = o3.rand_matrix()
rotated_point = torch.einsum('ij,zj->zi', rot, point)
rotated_result = torch.einsum('ij,j->i', rot, result)

distort = torch.randn(size = point.size())
factor = 1 / torch.mean(torch.norm(distort, dim = 1)) / 100  # ~ 0.1
distort *= factor

class Polynomial(torch.nn.Module):
    def __init__(self,
        node_rep: str = '1x0e',
        out_rep: str = '1x1o', 
        lmax: int = 2
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(node_rep)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.middle = o3.Irreps("5x0e + 5x0o + 5x1o")   # hidden states on the node
        self.irreps_out = o3.Irreps(out_rep)

        self.basis = 10  # because the distance is different, we need to encode base functions with NN

        self.normed_nonlinearity = NormActivation(
            self.middle, torch.sigmoid, bias = True
        )

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in,
            irreps_in2=self.irreps_sh,
            irreps_out=self.middle,
            internal_weights = False,
            shared_weights = False,
        )

        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=self.middle,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_out,
            internal_weights = True
        )

        self.fullyconnected = FullyConnectedNet(
            [self.basis, 50] + [self.tp1.weight_numel],
            torch.nn.functional.silu
        )

    def forward(self, pos, features) -> torch.Tensor:

        edge_from, edge_to = radius_graph(
            x=pos,
            r=2.5
        )
        edge_vec = pos[edge_to] - pos[edge_from]

        edge_sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=True,  # here we don't normalize otherwise it would not be a polynomial
            normalization='component'
        )

        dist_embedding = soft_one_hot_linspace(
                    x = edge_vec.norm(dim=1), 
                    start = 0, 
                    end = 3, 
                    number = self.basis, 
                    basis = 'smooth_finite', cutoff=True
        )

        tp_weights = self.fullyconnected(dist_embedding)
        node_features = self.tp1(features[edge_to], edge_sh, tp_weights)
        node_features = scatter(node_features, edge_from, dim = 0)
        node_features = self.normed_nonlinearity(node_features)
        node_features = self.tp2(node_features[edge_to], edge_sh)
        node_features = scatter(node_features, edge_from, dim = 0)

        return torch.sum(node_features, dim = 0)

f = Polynomial()
optim = torch.optim.Adam(f.parameters(), lr=1)
print(f)
    
for step in range(100):
    optim.zero_grad()
    pred = f(point, node_feature)
    loss = (pred - result).pow(2).sum()
    loss.backward()
    optim.step()

print("The network is fitted with the unrotated points:")
print("output vector:")
print(f(point, node_feature).detach().numpy())
print("target vector:")
print(result.numpy())

print("for a rotated points")
print("output vector:")
print(f(rotated_point, node_feature).detach().numpy())
print("target vector:")
print(rotated_result.numpy())

# basically, this network learns nothing except a direct mapping, it cannot treat
# distortion at all
print("")
print("the linear fit learns the direct fit, not mapping")
print("Distorted original point set")
print("output vector:")
print(f(point + distort, node_feature).detach().numpy())

print("Further distort this rotated point set")
print("output vector:")
print(f(rotated_point + distort, node_feature).detach().numpy())
