from e3nn.o3 import Irrep, rand_matrix
import torch
from numpy import pi

# for permutation, we can permute the first four 
point = torch.tensor([  [ 1.0, 1.0, 1.0],
                        [-1.0,-1.0, 1.0],
                        [ 1.0,-1.0,-1.0],
                        [-1.0, 1.0,-1.0],
                        [ 0.0, 0.0, 0.0]])

node_feature = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]])
result = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

#rot_angle = [torch.tensor(0.0, dtype = float), torch.tensor(pi/4, dtype = float), torch.tensor(0.0, dtype = float)]
#rot = Irrep('1o').D_from_angles(*rot_angle)
#rot_d = Irrep('2e').D_from_angles(*rot_angle)

rot_angle = rand_matrix()
rot = Irrep('1o').D_from_matrix(rot_angle)
rot_d = Irrep('2e').D_from_matrix(rot_angle)

rotated_point = torch.einsum('ij,zj->zi', rot.float(), point)
rotated_result = torch.einsum('ij,j->i', rot_d.float(), result)
