import plotly.graph_objects as go

from gbf import molecularfuntion
import numpy as np

grid_density = 50

def get_meshgrid(positions: np.array, grid_density: int = 50):
    extra_space = 1.5
    range_max = np.amax(positions, axis = 0)
    range_min = np.amin(positions, axis = 0)
    center = ( range_max + range_min ) / 2
    delta = (1 + extra_space) * np.max(( range_max - range_min )) / 2
    linear = [ 
        np.arange(center[i]-delta, center[i]+delta, delta * 2 / grid_density) for i in range(3)
    ]
    return np.meshgrid(*linear)

def plot_density(xs, ys, zs, value, filename, iso = 0.3):
    fig= go.Figure(data=go.Isosurface(
        x=xs.flatten(),
        y=ys.flatten(),
        z=zs.flatten(),
        value=value.flatten(),
        isomin = -iso,
        isomax =  iso,
        caps=dict(x_show=False, y_show=False, z_show = False)
    ))

    fig.write_html(filename)

if __name__ == "__main__":
    positions = np.array([[2,0,0],[-2,0,0],[0,2,0],[0,-2,0]])
    coefficients = np.array([[1.0, 0.0, 0.0, 0.0], 
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0], ])

    xs, ys, zs = get_meshgrid(positions)
    value = molecularfuntion(positions, coefficients, xs, ys, zs)
    plot_density(xs, ys, zs, value, "mo.html")