from automaticTB.linear_combination import LinearCombination, Site
from automaticTB.plotting.plot_LC import make_plot_normalized_LC
import numpy as np

sites = [
    Site(1, np.array([ 0, 0, 0])),
    Site(1, np.array([ 1, 1, 1])),
    Site(1, np.array([ 1,-1, 1])),
    Site(1, np.array([-1, 1, 1])),
    Site(1, np.array([-1,-1, 1])),
    Site(1, np.array([ 1, 1,-1])),
    Site(1, np.array([ 1,-1,-1])),
    Site(1, np.array([-1, 1,-1])),
    Site(1, np.array([-1,-1,-1]))
]

coefficients = np.eye(9)

linear_coefficients = LinearCombination(sites, coefficients)

def run_plot_all_single_function():
    make_plot_normalized_LC(linear_coefficients, "single_function.pdf")

run_plot_all_single_function()