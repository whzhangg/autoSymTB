from DFTtools.tools import read_generic_to_cpt
from DFTtools.DFT.output_parser import parse_pdos_folder
import os
import numpy as np
from mldos.data.milad_utility import Description

from torch_geometric.data import Data
import torch

from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.util.test import assert_equivariant

ef = 12.28
folder = "rotated_ones"
one_hot = {
    27: [1,0],
    51: [0,1]
}
torch_dtype = torch.float


def get_dataset():
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
            types = [ description.std_types[n.index] for n in neighbors[iatom - 1] ]
            pos = [ n.rij for n in neighbors[iatom - 1] ]
            x = torch.tensor([one_hot[i] for i in types], dtype=torch_dtype)
            pos = torch.tensor(np.array(pos), dtype=torch_dtype)
            occ = torch.tensor(occ, dtype=torch_dtype)

            edges = []
            for i in range(len(types)):
                for j in range(len(types)):
                    if types[i] == 27 and types[j] == 51:
                        edges.append([i,j])
                        edges.append([j,i])
            edge_index = torch.tensor(edges, dtype=torch.long)

            list_of_data.append(Data(x=x, pos = pos, y = occ, edge_index = edge_index))

    return list_of_data


class myInvariantPolynomial(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps('2x0e')
        self.irreps_sh = o3.Irreps.spherical_harmonics(3)
        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_out = o3.Irreps("1x2e")

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid,
            internal_weights = True
        )
        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=irreps_mid,
            irreps_out=irreps_out,
            internal_weights = True
        )
        self.irreps_out = self.tp2.irreps_out

    def forward(self, data) -> torch.Tensor:
        num_neighbors = 6  # typical number of neighbors
        edge_from = data.x[:,0] > 0.5
        edge_to = data.x[:,1] > 0.5

        edge_vec = data.pos[edge_to] - data.pos[edge_from]
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=False,  # here we don't normalize otherwise it would not be a polynomial
            normalization='component'
        )
        # For each node, the initial features are the sum of the spherical harmonics of the neighbors
        #node_features = scatter(edge_sh, edge_dst, dim=0).div(num_neighbors**0.5)


        # For each edge, tensor product the features on the source node with the spherical harmonics
        node_features0 = data.x[edge_to]

        self_features = self.tp1(node_features0, edge_sh)
        self_features = torch.sum(self_features,dim = 0).div(num_neighbors**0.5)

        node_features = self.tp2(self_features, self_features)

        return node_features
        

class InvariantPolynomial(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(3)
        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_out = o3.Irreps("1x2o")

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_sh,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid,
        )
        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_out,
        )
        self.irreps_out = self.tp2.irreps_out

    def forward(self, data) -> torch.Tensor:
        num_neighbors = 2  # typical number of neighbors
        num_nodes = 4  # typical number of nodes

        edge_src, edge_dst = radius_graph(
            x=data.pos,
            r=4,
            batch=data.batch
        )  # tensors of indices representing the graph
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=False,  # here we don't normalize otherwise it would not be a polynomial
            normalization='component'
        )

        # For each node, the initial features are the sum of the spherical harmonics of the neighbors
        node_features = scatter(edge_sh, edge_dst, dim=0).div(num_neighbors**0.5)

        # For each edge, tensor product the features on the source node with the spherical harmonics
        edge_features = self.tp1(node_features[edge_src], edge_sh)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        edge_features = self.tp2(node_features[edge_src], edge_sh)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        # For each graph, all the node's features are summed
        return scatter(node_features, data.batch, dim=0).div(num_nodes**0.5)


def main():
    from network import molecular_network
    net = molecular_network('2x0e','1x2o',
        lmax = 2,
        nlayer= 1,
        hidden_multiplicity = 20,
        embedding_nbasis = 10,
        linear_num_hidden = 50,
        linear_num_layer = 1,
        rcut = 3,
        num_neighbors = 3,
        use_alpha=False,
        skip_connection=False
        )

    
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    data = get_dataset()

    labels = torch.vstack([da.y for da in data])

    traindata = DataLoader(data, batch_size=int(len(data)/4))

    print(net)

    for step in range(500):
        for train in traindata:
            optim.zero_grad()
            pred = net(train)
            print(pred.shape)
            print(labels.shape)
            loss = (pred - labels).pow(2).sum()
            loss.backward()
            optim.step()

        if step % 10 == 0:
            print(pred)
            print(f"epoch {step:5d} | loss {loss:<10.1f}")
        


    # == Train ==
    '''
    for step in range(200):
        pred = f(data)
        loss = (pred - labels).pow(2).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            accuracy = pred.round().eq(labels).all(dim=1).double().mean(dim=0).item()
            print(f"epoch {step:5d} | loss {loss:<10.1f} | {100 * accuracy:5.1f}% accuracy")

    # == Check equivariance ==
    # Because the model outputs (psuedo)scalars, we can easily directly
    # check its equivariance to the same data with new rotations:
    print("Testing equivariance directly...")
    rotated_data, _ = tetris()
    error = f(rotated_data) - f(data)
    print(f"Equivariance error = {error.abs().max().item():.1e}")

    print("Testing equivariance using `assert_equivariance`...")
    # We can also use the library's `assert_equivariant` helper
    # `assert_equivariant` also tests parity and translation, and
    # can handle non-(psuedo)scalar outputs.
    # To "interpret" between it and torch_geometric, we use a small wrapper:

    def wrapper(pos, batch):
        return f(Data(pos=pos, batch=batch))

    # `assert_equivariant` uses logging to print a summary of the equivariance error,
    # so we enable logging
    logging.basicConfig(level=logging.INFO)
    assert_equivariant(
        wrapper,
        # We provide the original data that `assert_equivariant` will transform...
        args_in=[data.pos, data.batch],
        # ...in accordance with these irreps...
        irreps_in=[
            "cartesian_points",  # pos has vector 1o irreps, but is also translation equivariant
            None,  # `None` indicates invariant, possibly non-floating-point data
        ],
        # ...and confirm that the outputs transform correspondingly for these irreps:
        irreps_out=[f.irreps_out],
    )
    '''

main()