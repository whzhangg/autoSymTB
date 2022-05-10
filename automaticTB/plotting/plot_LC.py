import numpy as np
import matplotlib.pyplot as plt
import typing
from scipy.special import sph_harm
from ..linear_combination import AOLISTS, LinearCombination, Site


def sh_functions(l, m, theta, phi):
    # theta in 0-2pi; phi in 0-pi
    Y = sph_harm(abs(m), l, theta, phi)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    # real spherical harmonic
    return Y.real

def precompute_data() -> typing.Tuple[np.ndarray, np.ndarray]:
    # each of the wavefunction is well defined, so we precompute and store the data
    # Grids of polar and azimuthal angles
    density = 100
    theta = np.linspace(0, 2 * np.pi, 2 * density)
    phi = np.linspace(0, np.pi, density)
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    unit_vector = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
        ])

    computed_y = []
    for ao in AOLISTS:
        y = sh_functions(ao.l, ao.m, theta, phi)
        computed_y.append(y)

    computed_ys = np.stack(computed_y) 
    return unit_vector, computed_ys

unit_vector, computed_ys = precompute_data()

def combined_sh(coefficient):
    assert len(coefficient) == len(computed_ys)
    y_with_coefficients = []
    for co, computed_y in zip(coefficient, computed_ys):
        y_with_coefficients.append(co * computed_y)
    return np.sum(np.array(y_with_coefficients), axis = 0)

def on_site_xyz_color(coefficient, pos: np.ndarray):
    y = combined_sh(coefficient)
    scaled_vector = np.abs(y) * unit_vector
    
    vector_shape = scaled_vector.shape[1:]
    shift_value = np.tile(pos[:, np.newaxis, np.newaxis], vector_shape)
    assert shift_value.shape == scaled_vector.shape
    shifted_scaled = scaled_vector + shift_value
    color = y
    return shifted_scaled, color

def plot_vectors(combination: LinearCombination, ax, ax_lim):
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    for site, coefficient in zip(combination.sites, combination.coefficients):
        pos = site.pos
        vector, color = on_site_xyz_color(coefficient, pos)
        Yx, Yy, Yz = vector

        ax.plot_surface(Yx, Yy, Yz,
                        facecolors=cmap.to_rgba(color),
                        rstride=2, cstride=2)

    # Draw a set of x, y, z axes for reference.
    ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #ax.axis('off')
    ax.set_box_aspect((ax_lim * 2.2, ax_lim * 2.2, ax_lim * 2.2))

def get_limts(sites: typing.List[Site]) -> float:
    maximum = -1
    for site in sites:
        max_abs = max(np.abs(site.pos))
        maximum = max(maximum, max_abs)
    return maximum

def make_plot_normalized_LC(linearcombination: LinearCombination, filename: str = "linear_combination"):
    # it is not very efficient
    overhead = 1.0
    limit = overhead + get_limts(linearcombination.sites)

    if not linearcombination.is_normalized:
        linearcombination = linearcombination.normalized()

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    plot_vectors(linearcombination, ax, limit)

    fig.savefig(filename)

