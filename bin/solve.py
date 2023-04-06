#!/usr/bin/env python

"""
This script is a callable that retrives the space group character table
"""

import argparse

from automaticTB import tools
from automaticTB.solve.functions.solve import solve_interaction

if __name__ == "__main__":
    description = f"Solve the symmetry relationship between tight-binding parameters"
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument(
        "configfile", type = str, help = "a yaml file containing problem specification")
    args = parser.parse_args()
    
    config = tools.read_yaml(args.configfile)

    solve_interaction(**config)
