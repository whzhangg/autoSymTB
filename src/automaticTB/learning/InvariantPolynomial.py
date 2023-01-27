import torch, typing
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, FullTensorProduct
from e3nn.nn import Gate, FullyConnectedNet, NormActivation
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from .torch_parameters import torch_float
from torch_geometric.data import Data
from .MOadoptor import embed_atomic_orbital_coefficients


class InvariantPolynomial(torch.nn.Module):
    def __init__(self,
        node_rep: str = '1x0e',
        out_rep: str = '5x0e', 
        lmax: int = 3
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(node_rep)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.middle = o3.Irreps("5x0e + 5x1o + 5x2e + 5x3o")
        self.irreps_out = o3.Irreps(out_rep)

        self.basis = 20

        self.normed_nonlinearity = NormActivation(
            self.middle, torch.sigmoid, bias = True
        )

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_out,
            internal_weights = False,
            shared_weights = False,
        )

        self.tp2 = FullTensorProduct(
            irreps_in1= self.irreps_in,
            irreps_in2= self.irreps_out,
        )


        self.fullyconnected = FullyConnectedNet(
            [self.basis, 30] + [self.tp1.weight_numel],
            torch.nn.functional.silu
        )

    def forward(self, data) -> torch.Tensor:

        edge_sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=data.pos,
            normalize=True,
            normalization='component'
        )
        #dist_embedding = soft_unit_step(data.pos.norm(dim=1))
        dist_embedding = soft_one_hot_linspace(
                    x = data.pos.norm(dim=1), 
                    start = 0, 
                    end = 3.5, 
                    number = self.basis, 
                    basis = 'gaussian', cutoff=True
        ) # so that point at origin will have nonzero embedding

        print(dist_embedding)
        tp_weights = self.fullyconnected(dist_embedding)
        h_right = self.tp1(data.x_right, edge_sh, tp_weights)
        print(edge_sh, tp_weights)
        print(h_right)
        product = self.tp2(data.x_left[data.origin_index], h_right)
        result = torch.sum(product)
        return result


def get_fitter(list_Data: typing.List[Data], target: typing.List[float]):
    input_irrep = embed_atomic_orbital_coefficients.irreps_str
    net = InvariantPolynomial(
        node_rep = input_irrep, out_rep= '5x0e', lmax=3
    )
    optim = torch.optim.Adam(net.parameters(), lr=1e-1)
    
    for step in range(100):
        optim.zero_grad()
        loss = torch.tensor(0, dtype=torch_float)
        for data,y in zip(list_Data, target):
            pred = net(data)
            loss += (pred - y).pow(2).sum()
        loss.backward()
        optim.step()

    for data, y in zip(list_Data, target):
        pred = net(data)
        print(y, pred.item())
    
    return net