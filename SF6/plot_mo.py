import plotly.graph_objects as go

from gbf import wavefunction, molecularfuntion
import numpy as np

positions = np.array([[2,0,0],[-2,0,0],[0,2,0],[0,-2,0]])
coefficients = np.array([[1.0, 0.0, 0.0, 0.0], 
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 1.0], ])

lin = np.arange(-5, 5, 0.2)
xs, ys, zs = np.meshgrid(lin, lin, lin)
value = molecularfuntion(positions, coefficients, xs, ys, zs)

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
    plot_density(xs, ys, zs, value, "mo.html")