import numpy as np
import matplotlib.pyplot as plt
import typing
from scipy.special import sph_harm
from ..SALCs.linear_combination import LinearCombination
from ..atomic_orbitals import Orbitals
from ..structure import Site


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

    computed_y_dict = {}

    for ao in Orbitals([0,1,2]).sh_list:
        y = sh_functions(ao.l, ao.m, theta, phi)
        computed_y_dict[ao] = y

    return unit_vector, computed_y_dict


unit_vector, computed_y_dict = precompute_data()


def on_site_xyz_color(coefficient: np.ndarray, orb: Orbitals, pos: np.ndarray):
    # with coefficient
    y = np.zeros_like(computed_y_dict[orb.sh_list[0]])
    for co, orb in zip(coefficient, orb.sh_list):
        y += co * computed_y_dict[orb]

    scaled_vector = np.abs(y) * unit_vector
    
    vector_shape = scaled_vector.shape[1:]
    shift_value = np.tile(pos[:, np.newaxis, np.newaxis], vector_shape)
    assert shift_value.shape == scaled_vector.shape
    shifted_scaled = scaled_vector + shift_value
    color = y
    return shifted_scaled, color

def plot_vectors(combination: LinearCombination, ax, ax_lim):
    real_coefficients = combination.coefficients.real if np.iscomplex(combination.coefficients) else combination.coefficients
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    orbital_atom_slice = combination.orbital_list.atomic_slice_dict
    for i, site in enumerate(combination.sites):
        coefficient_positions = orbital_atom_slice[i]
        coeff = real_coefficients[coefficient_positions]
        
        vector, color = on_site_xyz_color(coeff, combination.orbital_list.orbital_list[i], site.pos)
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
        linearcombination = linearcombination.get_normalized()

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    plot_vectors(linearcombination, ax, limit)

    fig.savefig(filename)

