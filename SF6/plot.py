import plotly.graph_objects as go

from gbf import wavefunction
import numpy as np

orbit = "dx2-y2"

lin = np.arange(-5, 5, 0.2)
xs, ys, zs = np.meshgrid(lin, lin, lin)
value = wavefunction(1, orbit, (1,1,0), xs , ys, zs)
print(value.shape)
print(np.average(value))
fig= go.Figure(data=go.Isosurface(
    x=xs.flatten(),
    y=ys.flatten(),
    z=zs.flatten(),
    value=value.flatten(),
    isomin=-0.3,
    isomax=0.3,
    caps=dict(x_show=False, y_show=False, z_show = False)
))

fig.write_html(f"{orbit}.html")