from typing import Dict, Union

import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.v2103.gate_points_message_passing import MessagePassing


__all__ = ["OrbitalNetwork", "get_nearest_neighbor_orbital_mapper"]


class OrbitalNetwork(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_out,
        max_radius,
        num_neighbors,
        num_nodes,
        mul=25,
        layers=2,
        lmax=2,
        pool_nodes=True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing(
            irreps_node_input=irreps_node_input,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        # Create graph
        edge_index = radius_graph(data["pos"], self.max_radius, batch, max_num_neighbors=len(data["pos"]) - 1)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        return batch, data["x"], data["z"], edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        batch, node_inputs, node_attr, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data

        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        # node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)

        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            return scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return node_outputs


def get_nearest_neighbor_orbital_mapper(
    nele: int,
    feature_irreps: str,
) -> OrbitalNetwork:
    attribute_irreps = f"{nele}x0e"
    return OrbitalNetwork(
        feature_irreps, 
        attribute_irreps,
        "1x0e",
        max_radius = 5.0,
        num_neighbors = 1.0,
        num_nodes = 2.0
    )