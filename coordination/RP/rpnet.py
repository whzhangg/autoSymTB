import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn.o3 import FullyConnectedTensorProduct, Irreps, spherical_harmonics
from e3nn.nn import FullyConnectedNet, NormActivation
from e3nn.math import soft_one_hot_linspace
from itertools import permutations


class Polynomial(torch.nn.Module):
    def __init__(self,
        node_rep: str = '1x0e',
        out_rep: str = '1x2e', 
        lmax: int = 3,
        n_node: int = 5,
        pool: bool = False
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(node_rep) + Irreps(f'{n_node}x0e').simplify()
        self.irreps_sh = Irreps.spherical_harmonics(lmax)
        self.middle = Irreps("5x0e + 5x0o + 5x1o + 5x2e")   # hidden states on the node
        self.irreps_out = Irreps(out_rep)
        self.n_node = n_node

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

        result = torch.zeros(size = (self.irreps_out.dim,), dtype = features.dtype)

        additional = torch.eye(len(features), dtype = features.dtype)

        for per in list(permutations(range(self.n_node - 1)))[0:5]:
            permute = tuple([*per, self.n_node - 1])
            add = additional[permute, :]
            extended_features = torch.hstack((features, add))

            edge_from, edge_to = radius_graph(
                x=pos,
                r=3.0
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
            node_features = self.tp1(extended_features[edge_to], edge_sh, tp_weights)
            node_features = scatter(node_features, edge_from, dim = 0)
            node_features = self.normed_nonlinearity(node_features)
            node_features = self.tp2(node_features[edge_to], edge_sh)
            node_features = scatter(node_features, edge_from, dim = 0)
            result += torch.sum(node_features, dim = 0)
            
        result /= 24
        return result