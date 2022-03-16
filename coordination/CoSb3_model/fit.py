
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

ef = 12.28  # we intergrate to here to obtain the orbital occupation
folder = "rotated_ones"
one_hot = {
    27: [1,0],  # Co
    51: [0,1]   # Sb
}
torch_dtype = torch.float

def make_dataset():
    from DFTtools.DFT.output_parser import parse_pdos_folder
    from mldos.data.milad_utility import Description
    list_of_data = []
    for i in range(1,11):
        folder_rot = os.path.join(folder, f"rot{i}")
        scf = os.path.join(folder_rot, 'scf.in')

        dos_result = parse_pdos_folder(folder_rot)
        occupation = {}
        for iatom, dos in dos_result.items():
            energy = dos['3d']['energy']
            mask = energy < ef
            result = []
            for orb in ['pdos_1','pdos_2','pdos_3','pdos_4','pdos_5']:
                d = dos['3d'][orb]
                result.append( np.sum(d[mask]) * 0.05 )
            occupation[iatom] = result

        allpoints = [iatom-1 for iatom in occupation.keys()]
        description = Description(filename=scf, keepstructure=True)
        neighbors = description.get_neighbors(allpoints, bonded=False, firstshell=True, cutoff=5)

        for iatom, occ in occupation.items():
            pos = [ n.rij for n in neighbors[iatom - 1] ]

            x = torch.tensor([[1.0]] * len(pos), dtype=torch_dtype)
            pos = torch.tensor(np.array(pos), dtype=torch_dtype)
            occ = torch.tensor(occ, dtype=torch_dtype)

            list_of_data.append(Data(x=x, pos = pos, y = occ))

    return list_of_data


class InvariantPolynomial(torch.nn.Module):
    def __init__(self,
        node_rep: str = '1x0e',
        out_rep: str = '1x2e', 
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
            [self.basis, 30] + [self.tp1.weight_numel],
            torch.nn.functional.silu
        )

    def forward(self, data) -> torch.Tensor:
        num_neighbors = 6 
        datax = data.x

        edge_to = data.edge_index[:,1]
        Co = torch.mode(edge_to)[0]

        edge_from, edge_to = radius_graph(
            x=data.pos,
            r=3.5
        )
        edge_vec = data.pos[edge_to] - data.pos[edge_from]
        Co = torch.mode(edge_to)[0]

        edge_sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=True,  # here we don't normalize otherwise it would not be a polynomial
            normalization='component'
        )

        dist_embedding = soft_one_hot_linspace(
                    x = edge_vec.norm(dim=1), 
                    start = 0, 
                    end = 3.5, 
                    number = self.basis, 
                    basis = 'smooth_finite', cutoff=True
        )

        tp_weights = self.fullyconnected(dist_embedding)
        node_features = self.tp1(datax[edge_to], edge_sh, tp_weights)
        node_features = scatter(node_features, edge_from, dim = 0).div(num_neighbors**0.5)

        node_features = self.tp2(node_features[edge_to], edge_sh)
        node_features = scatter(node_features, edge_from, dim = 0).div(num_neighbors**0.5)
        
        return node_features[Co]


def perturb_input(data, mode:str):
    if mode == "displace":
        # for displace, learning rate would be 1e-1, this achieve almost perfect fitting
        perturb = torch.randn(size = data.pos.size(), dtype = data.pos.dtype)
        factor = 1 / torch.mean(torch.norm(perturb, dim = 1)) / 50  # ~ 0.1
        data.pos = data.pos + perturb * factor
        return data, '1x0e'

    elif "random_feature=" in mode:
        # it seems that random feature does not really work (1x0e)
        # why single random feature is not working?
        # but if I use random (2x0e) feature, the fitting is perfect
        nrandom = int(mode.split("=")[1])
        perturb = torch.randn(size = (data.x.size()[0], nrandom), dtype = data.x.dtype)
        data.x = perturb
        return data, f'{nrandom}x0e'

    elif mode == "extend":
        # however, simply extend feature to (2x0e) filled with 1.0 does not 
        # work
        datax = torch.hstack((data.x, data.x))
        data.x = datax
        return data, '2x0e'

    elif mode == "onehot":
        # this achieve almost perfect fitting
        datax = torch.eye(data.x.size()[0], dtype = data.x.dtype)
        data.x = datax
        return data, f'{len(datax)}x0e'

    elif mode == "vector":
        vector = torch.tensor([-0.2, 0.53, 0.09 ], dtype = data.x.dtype)
        vectors = torch.vstack(len(data.x) * [vector])
        data.x = torch.hstack([data.x, vectors])
        return data, '1x0e + 3x0e'

    elif "color=" in mode:
        # we mark the first n node different color
        # this also does not work as expected
        n = int(mode.split("=")[1])
        datax = torch.zeros(size = (len(data.x), 2), dtype = data.x.dtype)
        for i in range(datax.size()[0]):
            if i < n:
                datax[i, 0] = 1.0
            else:
                datax[i, 1] = 1.0
        data.x = datax
        return data, '2x0e'


def test():
    storage_file = f"{folder}.pt"
    if os.path.exists(storage_file):
        datalist = torch.load(storage_file)
    else:
        datalist = make_dataset()
        torch.save(datalist, storage_file)
    sh = o3.Irreps.spherical_harmonics(2)
    data = datalist[1]
    full = FullTensorProduct(
            irreps_in1= '1x0e+1x1o',
            irreps_in2= sh
        )
    print(full)

    vector = data.pos[1]
    vectors = torch.vstack(len(data.x) * [vector])
    data.x = torch.hstack([data.x, vectors])

    edge_from, edge_to = radius_graph(
            x=data.pos,
            r=3.0
    )
    edge_vec = data.pos[edge_to] - data.pos[edge_from]



    edge_sh = o3.spherical_harmonics(
            l=sh,
            x=edge_vec,
            normalize=True,  # here we don't normalize otherwise it would not be a polynomial
            normalization='component'
    )

    feature = full(data.x[edge_to], edge_sh)
    print(torch.sum(feature, dim = 0))

def main():
    storage_file = f"{folder}.pt"
    print(os.path.exists(storage_file))
    if os.path.exists(storage_file):
        datalist = torch.load(storage_file)
    else:
        datalist = make_dataset()
        torch.save(datalist, storage_file)

    x = datalist[1]
    labels = x.y

    x, irrep_node = perturb_input(x, 'random_feature=1')
    print(x.x)

    net = InvariantPolynomial(node_rep=irrep_node, out_rep='1x2e')
    print(net)

    optim = torch.optim.Adam(net.parameters(), lr=1e-1)
    
    for step in range(500):
        optim.zero_grad()
        pred = net(x)
        loss = (pred - labels).pow(2).sum()
        if step % 20 == 0:
            print(pred)
        loss.backward()
        optim.step()
    
    
    print(pred)
    print(x.y)

main()