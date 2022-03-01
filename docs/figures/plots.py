from mplconfig import get_acsparameter
import numpy as np
import matplotlib.pyplot as plt

def plot_shift(filename = "3dnn_shift.pdf"):
    with plt.rc_context(get_acsparameter(width = "single", n = 1, color = "line")):
        fig = plt.figure()
        axes = fig.subplots()

        x = np.arange(-6,6,0.2)
        y1 = np.exp(-x**2)
        y2 = np.exp(-(x-2)**2)

        axes.set_xlabel("$x$")
        axes.set_ylabel("feature map intensity")
        
        axes.set_xlim(-5, 5)
        axes.plot(x, y1, label = "f(x)")
        axes.plot(x, y2, label = "f(x-2)")
        axes.legend()

        fig.savefig(filename)

plot_shift()