#!/usr/bin/env python

"""
This script is a callable that retrives the space group character table
"""

import typing
import argparse

from automaticTB.solve import sitesymmetry


def get_sitesymmetry_groups() -> typing.Dict[str, sitesymmetry.SiteSymmetryGroup]:
    results = {}
    for group in sitesymmetry.GroupsList:
        results[group] = sitesymmetry.SiteSymmetryGroup.from_groupname(group)
    return results


def get_character_str(value: complex) -> str:
    if abs(value.imag) < 1e-6:
        return f"{value.real:.2f}"
    else:
        return f"{value:.2f}"


def get_sitesymmetrygroup_str(group: sitesymmetry.SiteSymmetryGroup) -> str:
    lines = []
    lines.append(f"group name : {group.groupname}")
    lines.append(f"order      : {len(group.operations)}")
    lines.append(f"subgroups  : " + " ".join([sg for sg in group.subgroups]))
    irrep_line      = "Irr. Representation"
    irrep_dimension = "Dimension          "
    for key, val in group.irrep_dimension.items():
        irrep_line += f"{key:>10s}"
        irrep_dimension += f"{val:>10d}"
    lines.append(irrep_line)
    lines.append(irrep_dimension)
    lines.append("")
    lines.append("Character Table")
    rows = {
        "Operations": list(group.dressed_op.keys())
    }
    for irrep, character in group.irreps.items():

        rows[irrep] = [ get_character_str(c) for c in character ]

    format = "{:>10s}" + "{:>15s}" * len(group.operations)
    for key, value_list in rows.items():
        lines.append(format.format(key, *value_list))
    
    return "\n".join(lines)


if __name__ == "__main__":
    all_groups = " ".join(sitesymmetry.GroupsList)
    description = f"Retrive the character table of a point group: {all_groups}"
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument("--symbol", type = str, help = "eg: mmm")

    # since group name contain -, it's easier to include as optional arguments
    # https://stackoverflow.com/questions/69925189/python-command-line-argument-string-starts-with
    args = parser.parse_args()
    g = get_sitesymmetry_groups()
    assert args.symbol in g.keys()
    data = g[args.symbol]
    print(
        get_sitesymmetrygroup_str(data)
    )

