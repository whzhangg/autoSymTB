# 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# The following import configures Matplotlib for 3D plotting.
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

plt.rc('text', usetex=True)

# Grids of polar and azimuthal angles
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
# Create a 2-D meshgrid of (theta, phi) angles.
theta, phi = np.meshgrid(theta, phi)
# Calculate the Cartesian coordinates of each point in the mesh.
xyz = np.array([np.sin(theta) * np.sin(phi),
                np.cos(theta),
                np.sin(theta) * np.cos(phi)])


def sh_functions(l, m, phi, theta):
    Y = sph_harm(abs(m), l, phi, theta)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    # real spherical harmonic
    return Y.real

def plot_Y(ax, el, m):
    """Plot the spherical harmonic of degree el and order m on Axes ax."""

    # NB In SciPy's sph_harm function the azimuthal coordinate, theta,
    # comes before the polar coordinate, phi.
    Y = sph_harm(abs(m), el, phi, theta)

    # Linear combination of Y_l,m and Y_l,-m to create the real form.
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
        
    Yx, Yy, Yz = np.abs(Y) * xyz # scales the vector


    # Colour the plotted surface according to the sign of Y.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(Y.real),
                    rstride=2, cstride=2)

    # Draw a set of x, y, z axes for reference.
    ax_lim = 0.5
    ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax.set_title(r'$Y_{{{},{}}}$'.format(el, m))
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')

def plot_3d_combined(coefficients, ax, ax_lim):
    coefficients = np.array(coefficients, dtype = float)
    norm = np.linalg.norm(coefficients)
    coefficients /= norm
    l = 2
    y_results = []
    for cm, m in zip(coefficients, [-2, -1, 0, 1, 2]):
        y_results.append(sh_functions(l, m, phi, theta) * cm)
    y_results = np.array(y_results)
    y_results = np.sum(y_results, axis = 0)

    Yx, Yy, Yz = np.abs(y_results) * xyz # scales the vector

    # Colour the plotted surface according to the sign of Y.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(y_results),
                    rstride=2, cstride=2)

    # Draw a set of x, y, z axes for reference.
    ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax.set_title('{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>7.2f}'.format(*coefficients))
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #ax.axis('off')

def plot_function(l, m):
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    plot_Y(ax, l, m)
    fig.savefig('Y{}_{}.png'.format(l, m))

def plot_rotates():
    import torch
    from e3nn.o3 import rand_matrix, Irrep
    irrepl2 = Irrep('2e')
    angle = torch.tensor([2, 0.8, 1], dtype = float)  # some rotation
    rotation = irrepl2.D_from_matrix(rand_matrix())
    print(rotation)
    dz_coefficient = np.array([0, 0, 1, 0, 0], dtype = float)
    rot_coefficient = torch.einsum('ij,j -> i', rotation.float(), torch.from_numpy(dz_coefficient).float() )
    rot_coefficient = rot_coefficient.detach().numpy()

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    plot_3d_combined(rot_coefficient, ax)
    fig.savefig('rotate4.png')


if __name__ == "__main__":
    plot_function(2, -2)
    plot_function(2, -1)
    plot_function(2, 0)
    plot_function(2, 1)
    plot_function(2, 2)

