from e3nn.o3 import Irrep, rand_matrix
import torch
from numpy import pi

point_original = torch.tensor([[ 1.0, 0.0, 0.0],
                      [ 0.0, 1.0, 0.0],
                      [ 0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [ 0.0,-1.0, 0.0],
                      [ 0.0, 0.0,-1.0]])
distort = torch.randn(size = point_original.size())
factor = 1 / torch.mean(torch.norm(distort, dim = 1)) / 50  # ~ 0.1
distort *= factor
point = point_original + distort

# point with some small distortion

point  = torch.tensor([ [ 0.9734,  0.0135,  0.0091],
                        [-0.0055,  0.9896,  0.0036],
                        [-0.0097, -0.0013,  0.9939],
                        [-0.9977, -0.0153,  0.0164],
                        [-0.0177, -0.9854,  0.0102],
                        [ 0.0127,  0.0118, -0.9981]]
                    )

node_feature = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
result = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])

rot_angle = [torch.tensor(0.0, dtype = float), torch.tensor(pi/4, dtype = float), torch.tensor(0.0, dtype = float)]
rot = Irrep('1o').D_from_angles(*rot_angle)
rot_d = Irrep('2e').D_from_angles(*rot_angle)

rot_angle = rand_matrix()
rot = Irrep('1o').D_from_matrix(rot_angle)
rot_d = Irrep('2e').D_from_matrix(rot_angle)

print(point)
rotated_point = torch.einsum('ij,zj->zi', rot.float(), point)
print(rotated_point)
rotated_result = torch.einsum('ij,j->i', rot_d.float(), result)
