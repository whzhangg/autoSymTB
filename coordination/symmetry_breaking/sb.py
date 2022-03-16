import torch
from e3nn.o3 import FullyConnectedTensorProduct, spherical_harmonics, Irreps, rand_matrix
from e3nn.nn import FullyConnectedNet

from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn.math import soft_one_hot_linspace

point = torch.tensor([  [ 1.0, 0.0, 0.0],
                        [ 0.0, 1.0, 0.0],
                        [ 0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [ 0.0,-1.0, 0.0],
                        [ 0.0, 0.0,-1.0],
                        [ 0.0, 0.0, 0.0]])

point = torch.tensor([[1.0,1.0,1.0],[-1.0,-1.0,1.0],[1.0,-1.0,-1.0],[-1.0,1.0,-1.0],[0.0,0.0,0.0]])

node_feature = torch.tensor([[1.0],[1.0],[1.0],[1.0],[1.0]])
vector = torch.randn((3,), dtype= node_feature.dtype)
vector = torch.vstack([vector] * len(node_feature))
#node_feature = torch.hstack([node_feature, vector])

#node_feature = torch.eye(len(point))
print(node_feature)
distort = torch.randn(size = point.size())
factor = 1 / torch.mean(torch.norm(distort, dim = 1)) / 50  # ~ 0.1
distort *= factor
point_distort = point

basis = 10

rotation = rand_matrix()
#point_distort = torch.einsum('ij,zj->zi', rotation, point_distort)

irreps_out = '1x0e + 1x1o + 1x2e'
irreps_sh = Irreps.spherical_harmonics(5)

edge_from, edge_to = radius_graph(
                x=point_distort,
                r=2.0
            )

edge_vec = point_distort[edge_to] - point_distort[edge_from]

edge_sh = spherical_harmonics(
                l=irreps_sh,
                x=edge_vec,
                normalize=True,  # here we don't normalize otherwise it would not be a polynomial
                normalization='component'
            )

dist_embedding = soft_one_hot_linspace(
                        x = edge_vec.norm(dim=1), 
                        start = 0, 
                        end = 2.0, 
                        number = basis, 
                        basis = 'smooth_finite', cutoff=True
            )


results = []
for i in range(50):
    full = FullyConnectedTensorProduct(
        '1x0e', irreps_sh, irreps_out, 
                internal_weights = False,
                shared_weights = False,
    )
    fullyconnected = FullyConnectedNet(
                [basis, 20] + [full.weight_numel],
                torch.nn.functional.silu
            )
    tmp = full(node_feature[edge_to], edge_sh, fullyconnected(dist_embedding))
    tmp = scatter(tmp, edge_from, dim = 0)
    r = tmp[-1]
    results.append(r)

results = torch.vstack(results)
print(torch.linalg.matrix_rank(results))

raise
import matplotlib.pyplot as plt
from angular_wavefunction import combined

fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(projection='3d')
combined(r.detach(), irreps_out , ax)
fig.savefig("combined.pdf")