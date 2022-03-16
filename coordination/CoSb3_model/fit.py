
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
            types = [ description.std_types[n.index] for n in neighbors[iatom - 1] ]
            pos = [ n.rij for n in neighbors[iatom - 1] ]

            x = torch.tensor([[1.0]] * len(pos), dtype=torch_dtype)
            pos = torch.tensor(np.array(pos), dtype=torch_dtype)
            occ = torch.tensor(occ, dtype=torch_dtype)

            edges = []
            for i in range(len(types)):
                for j in range(len(types)):
                    if types[i] == 27 and types[j] == 51:
                        edges.append([i,j])  # edges from Co -> Sb
                        edges.append([j,i])  # edges from Sb -> Co
                        
            edge_index = torch.tensor(edges, dtype=torch.long)

            list_of_data.append(Data(x=x, pos = pos, y = occ, edge_index = edge_index))

    return list_of_data


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class myNormedInvariantPolynomial(torch.nn.Module):
    def __init__(self,
        node_rep: str = '1x0e',
        out_rep: str = '1x2e', 
        lmax: int = 3
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(node_rep)
        #self.irreps_sh = o3.Irreps("1x0e + 1x2e")
        self.irreps_sh = o3.Irreps.spherical_harmonics(5)
        self.middle = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o + 1x4e")
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
            [self.basis, 50] + [self.tp1.weight_numel],
            torch.nn.functional.silu
        )

    def forward(self, data) -> torch.Tensor:
        num_neighbors = 6 
        datax = data.x

        edge_to = data.edge_index[:,1]
        Co = torch.mode(edge_to)[0]

        edge_from, edge_to = radius_graph(
            x=data.pos,
            r=4.0
        )
        edge_vec = data.pos[edge_to] - data.pos[edge_from]

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

        # For each edge, tensor product the features on the source node with the spherical harmonics
        #print(datax.shape)

        #print(node_features0.shape)
        #print(edge_sh.shape)

        #print(edge_sh.detach().numpy())
        #print(self.tp1.weight.detach().numpy())
        #print(tp_weights.shape)
        #print(edge_sh.shape)
        node_features = self.tp1(datax[edge_to], edge_sh, tp_weights)

        #print(node_features)
        #node_features = self.normed_nonlinearity(node_features)
        node_features = scatter(node_features, edge_from, dim = 0).div(num_neighbors**0.5)
        
        #full = FullTensorProduct(self.middle, self.irreps_sh)

        node_features = node_features[edge_to]
        #fullresult = full(node_features, edge_sh)
        #print(fullresult[0])
        node_features = self.tp2(node_features, edge_sh)
        #print(node_features)

        node_features = scatter(node_features, edge_from, dim = 0)
        
        return node_features[Co]


def main():
    storage_file = f"{folder}.pt"
    print(os.path.exists(storage_file))
    if os.path.exists(storage_file):
        datalist = torch.load(storage_file)
    else:
        datalist = get_dataset()
        torch.save(datalist, storage_file)
    
    #print(datalist[1].x)
    #print(datalist[1].pos)
    #print(datalist[1].edge_index)
    #print(datalist[1].y)

    net = myNormedInvariantPolynomial()
    print(net)

    #net(datalist[1])

    optim = torch.optim.Adam(net.parameters(), lr=1)
    labels = datalist[1].y

    
    for step in range(20):
        optim.zero_grad()
        pred = net(datalist[1])
        #print(pred)
        #print(labels)
        loss = (pred - labels).pow(2).sum()
        loss.backward()
        optim.step()
    print()
    #print(net(datalist[1]))
    print(pred)
    print(datalist[1].y)
    print()
    #print(net(datalist[2]))
    print(datalist[2].y)


def test_tp():
    irreps_sh = o3.Irreps.spherical_harmonics(2)
    middle = o3.Irreps("30x0e + 30x1e + 30x1o + 30x2e")
    irreps_out = o3.Irreps("1x2e")

    print(FullTensorProduct(irreps_sh, '1x0e'))
    print(FullyConnectedTensorProduct(middle, middle, '1x2e'))

if __name__ == "__main__":
    main()