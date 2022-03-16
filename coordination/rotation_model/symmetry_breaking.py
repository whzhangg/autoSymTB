# here I try to use Smidt's idea of introducing additional parameter for 
# symmetry breaking

from input_data import point, result, rotated_point, rotated_result, node_feature, point_original, rotated_point_original
from fit_function import Polynomial
import torch

from torch import nn
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn.o3 import FullyConnectedTensorProduct, Irreps, spherical_harmonics
from e3nn.nn import FullyConnectedNet, NormActivation
from e3nn.math import soft_one_hot_linspace

# this script illustrate that using a learnable global parameter appended to 
# each point, we can break the symmetry and produce the fitting correctly

class SymmetryBreakingPolynomial(torch.nn.Module):
    odd = {
        -1 : 'o',
         1 : 'e'
    }
    def __init__(self,
        node_rep: str = '1x0e',
        out_rep: str = '1x2e', 
        lmax: int = 3,
        breaking_order: int = 1
    ) -> None:
        super().__init__()
        self.symmtry_breaking = Irreps("1x{:d}{:s}".format(breaking_order, self.odd[-1**breaking_order]))

        self.irreps_in = Irreps(node_rep) + self.symmtry_breaking
        self.irreps_sh = Irreps.spherical_harmonics(lmax)
        self.middle = Irreps("5x0e + 5x0o + 5x1o + 5x2e")   # hidden states on the node
        self.irreps_out = Irreps(out_rep)

        print(self.irreps_in)

        self.sb_weight = nn.Parameter(torch.randn(size = (self.symmtry_breaking.dim,)))

        self.basis = 20  # because the distance is different, we need to encode base functions with NN

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
        features = torch.hstack([features, torch.vstack(len(features) * [self.sb_weight] )])

        edge_from, edge_to = radius_graph(
            x=pos,
            r=2.0
        )
        edge_vec = pos[edge_to] - pos[edge_from]

        edge_sh = spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=True,  # here we don't normalize otherwise it would not be a polynomial
            normalization='component'
        )

        dist_embedding = soft_one_hot_linspace(
                    x = edge_vec.norm(dim=1), 
                    start = 0, 
                    end = 2.0, 
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

f = SymmetryBreakingPolynomial(breaking_order=1)
optim = torch.optim.Adam(f.parameters(), lr=1e-1)
print(f)

point = point_original

for step in range(200):
    optim.zero_grad()
    pred = f(point, node_feature)
    loss = (pred - result).pow(2).sum()
    loss += torch.sum(torch.abs(f.sb_weight))
    print(torch.norm(f.sb_weight))
    loss += 1 - torch.norm(f.sb_weight)
    loss.backward()
    optim.step()

print(f.sb_weight)
print("The network is fitted with the unrotated points:")
print("output vector:")
print(f(point, node_feature).detach().numpy())
print("target vector:")
print(result.numpy())
