import numpy as np
import torch


from input_data import point, result, rotated_point, rotated_result, node_feature, point_original
from fit_function import Polynomial
from angular_wavefunction import plot_3d_combined

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points_wavefunction(pts, coefficients, filename):
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    plot_3d_combined(coefficients, ax, ax_lim=1.0)
    ax.scatter(pts[:,0]/1.3, pts[:,1]/1.3, pts[:,2]/1.3, marker = 'o')
    ax.set_box_aspect((np.ptp(pts[:,0]/1.2), np.ptp(pts[:,0]/1.2), np.ptp(pts[:,0]/1.2)))
    fig.savefig(filename)


f = Polynomial()
optim = torch.optim.Adam(f.parameters(), lr=1e-1)
print(f)
    
for step in range(300):
    optim.zero_grad()
    pred = f(point, node_feature)
    loss = (pred - result).pow(2).sum()
    loss.backward()
    optim.step()

print("The network is fitted with the unrotated points:")
print("output vector:")
print(f(point, node_feature).detach().numpy())
print("target vector:")
print(result.numpy())
plot_points_wavefunction(point.detach().numpy(), result.detach().numpy(), "unrotated.pdf")

print("for a rotated points")
print("output vector:")
print(f(rotated_point, node_feature).detach().numpy())
print("target vector:")
print(rotated_result.numpy())
plot_points_wavefunction(rotated_point.detach().numpy(), rotated_result.detach().numpy(), "rotated.pdf")

print("However, if the coordinates is exactly octahedral, the network cannot find this mapping, we need to somehow break the symmetry")
print("this can be varified by setting point = point_original before the network training")
#plot_points_wavefunction(point.detach().numpy(), result.detach().numpy(), "unrotated.pdf")