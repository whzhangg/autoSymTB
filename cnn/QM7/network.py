# 
# this script contain the e3nn network
#
from torch_scatter import scatter
import torch

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.math import soft_one_hot_linspace

def print_network(net):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class convolution(torch.nn.Module):
    def __init__(self,
        irreps_node,
        irreps_node_attributes,
        irreps_filter, 
        irreps_out,
        neurons,
        num_neighbor,
        use_alpha,
        skip_connection
    ) -> None:
        super().__init__()

        if use_alpha == True: skip_connection = True

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_node_attributes = o3.Irreps(irreps_node_attributes)
        self.irreps_filter = o3.Irreps(irreps_filter)
        self.irreps_out = o3.Irreps(irreps_out)
        self.skip_connection = skip_connection
        self.use_alpha = use_alpha

        if self.skip_connection:
            self.sc = o3.FullyConnectedTensorProduct(self.irreps_node, self.irreps_node_attributes, self.irreps_out)

        self.self_interaction = o3.FullyConnectedTensorProduct(self.irreps_node, self.irreps_node_attributes, self.irreps_node)


        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node):
            for j, (_, ir_edge) in enumerate(self.irreps_filter):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        assert irreps_mid.dim > 0, f"irreps_node_input={self.irreps_node} time irreps_edge_attr={self.irreps_filter} produces nothing in irreps_node_output={self.irreps_out}"

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]
        
        self.tp = o3.TensorProduct(
            self.irreps_node, 
            self.irreps_filter, 
            irreps_mid, 
            instructions,
            internal_weights = False,
            shared_weights = False
        )

        self.fullyconnected = FullyConnectedNet(
            neurons + [self.tp.weight_numel],
            torch.nn.functional.silu
        )

        self.linear_out = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attributes, self.irreps_out)

        if self.use_alpha:
            self.alpha = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attributes, '1x0e')
            with torch.no_grad():
                self.alpha.weight.zero_()
            assert self.alpha.output_mask[0] == 1.0, f"irreps_mid={irreps_mid} and irreps_node_attr={self.irreps_node_attributes} are not able to generate scalars"

        self.num_neighbors = num_neighbor

    def forward(self, node_input, node_attributes, edge_src, edge_dst, edge_attr, dist_embedding):
        
        self_connection = self.self_interaction(node_input, node_attributes)
        tp_weights = self.fullyconnected(dist_embedding)

        intermediate = self.tp(self_connection[edge_dst], edge_attr, tp_weights)
        intermediate = scatter(intermediate , edge_src, dim = 0, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)

        node_output = self.linear_out(intermediate, node_attributes)

        if self.skip_connection:
            skipped = self.sc(node_input, node_attributes)
            if self.use_alpha:
                alpha = self.alpha(intermediate, node_attributes)
                m = self.sc.output_mask
                alpha = (1 - m) + alpha * m
                node_output = node_output * alpha + skipped

            else:
                node_output = node_output + skipped

        return node_output


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class molecular_network(torch.nn.Module):

    """
    The tunable parameters of this network is:
    lmax, nlayer, rcut, embedding_nbasis, multiplicity of hidden representations
    The important parameters are:
    Model complexity: 
    lmax:    [2, 3]
    nlayer:  [2, 3, 4, 5]
    multiplicity: [20, 40, 60]

    original feature representation
    rcut:    [3, 4, 5]
    embedding_nbasis: [20, 30, 40]

    Linear layer for embedding:
    num_hidden = [100, 200, 300]
    layer_hidden = [1, 2, 3]
    """

    def __init__(self,
        node_input_rep,
        output_rep,
        lmax = 2,
        nlayer = 3, 
        hidden_multiplicity: int = 20,
        embedding_nbasis: int = 20,
        linear_num_hidden: int = 100,
        linear_num_layer: int = 1,
        rcut = 5.0,
        num_neighbors = 6.0,
        use_alpha: bool = False,
        skip_connection: bool = False
    ) -> None:
        super().__init__()

        self.num_neighbors = num_neighbors
        self.embedding_nbasis = embedding_nbasis
        self.rcut = rcut

        self.use_alpha = use_alpha
        self.skip_connection = skip_connection

        self.irreps_in = node_input_rep
        self.irreps_out = output_rep
        self.irreps_hidden = nlayer * [o3.Irreps([
            (hidden_multiplicity, (l, p))
            for l in range(lmax+1)
            for p in [-1, 1]
        ])]

        self.neurons = [ self.embedding_nbasis ] + [linear_num_hidden] * linear_num_layer
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        self.layers = torch.nn.ModuleList()

        input_irrps = self.irreps_in
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }
        for i, out_irrps in enumerate(self.irreps_hidden):

            irreps_scalars = o3.Irreps([
                (mul, ir)
                for mul, ir in out_irrps
                if ir.l == 0 and tp_path_exists(input_irrps, self.irreps_sh, ir)
            ]).simplify()
            irreps_gated = o3.Irreps([
                (mul, ir)
                for mul, ir in out_irrps
                if ir.l > 0 and tp_path_exists(input_irrps, self.irreps_sh, ir)
            ])
            if irreps_gated.dim > 0:
                if tp_path_exists(input_irrps, self.irreps_sh, "0e"):
                    ir = "0e"
                elif tp_path_exists(input_irrps, self.irreps_sh, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(f"irreps_node={input_irrps} times irreps_edge_attr={self.irreps_sh} is unable to produce gates needed for irreps_gated={irreps_gated}")
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = convolution(input_irrps, '0e', self.irreps_sh, gate.irreps_in,
                                self.neurons, self.num_neighbors, 
                                self.use_alpha, self.skip_connection)

            self.layers.append(Compose(conv, gate))
            input_irrps = gate.irreps_out

        conv = convolution(input_irrps, '0e', self.irreps_sh, self.irreps_out,
                                self.neurons, self.num_neighbors,
                                self.use_alpha, self.skip_connection)

        self.layers.append(conv)

        # create all the components

    def preprocess(self, data):
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]

        edge_vec = data.pos[edge_dst] - data.pos[edge_src]
        edge_attr = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=True,
            normalization='component'
        )
        #print(edge_attr.shape)

        dist_embedding = soft_one_hot_linspace(
                    x = edge_vec.norm(dim=1), 
                    start = 0, 
                    end = self.rcut, 
                    number = self.embedding_nbasis, 
                    basis = 'smooth_finite', cutoff=True
        )
        #print(dist_embedding.shape)
        node_attr = torch.ones(data.x.shape[0], 1).to(data.x.device)

        return data.batch, data.x, node_attr, edge_src, edge_dst, edge_attr, dist_embedding

    def forward(self, data) -> torch.Tensor:
        batch, node_input, node_attr, edge_src, edge_dst, edge_attr, dist_embedding \
            = self.preprocess(data)

        for i, layer in enumerate(self.layers):
            node_input = layer(node_input, node_attr, edge_src, edge_dst, edge_attr, dist_embedding)

        return scatter(node_input, batch, dim=0, dim_size=len(data.y))


if __name__ == '__main__':
    from load_data import QM7
    from torch_geometric.loader import DataLoader
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    dataset = QM7(rcut=5.0)
    loader = DataLoader(dataset, batch_size=32)
    net = molecular_network('5x0e','1x0e', nlayer=2)
    net.to(device)
    print(net)
    for batch in loader:
        net(batch)
        break
    print("no problem")