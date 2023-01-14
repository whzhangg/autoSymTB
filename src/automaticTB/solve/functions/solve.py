import typing, os, time

import numpy as np
from automaticTB.tools import read_cif_to_cpt
from automaticTB.solve.structure import Structure
from automaticTB.solve.interaction import (
    get_InteractingAOSubspaces_from_cluster, CombinedAOSubspaceInteraction
)
from automaticTB.interface import OrbitalPropertyRelationship


def solve_interaction(
    structure: typing.Union[str, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]],
    orbitals_dict: typing.Dict[str, str],
    rcut: typing.Optional[float] = None,
    standardize: bool = True,
    find_additional_symmetry: bool = False,
    save_filename: str = None
) -> None:
    """
    this function wraps the process of finding the unique interaction parameter. It return `None` but will print the output log information in a similar fashion of traditional calculation 
    program. It will write a file which can be read to perform extra operation/property extraction. 
    """

    print(f"Program start @ " + time.strftime(r"%Y/%m/%d %H:%M:%S", time.localtime()))
    print("")
    if type(structure) == str:
        c, p, t = read_cif_to_cpt(structure)
        print(f"# Reading crystal structure from {structure}")
    else:
        print(f"# Using given cell, positions and types")
        c, p, t = structure
    structure = Structure(c, p, t, orbitals_dict, rcut, standardize)

    print("")
    structure.print_log()
    print("")

    print("# Solving interaction for atom-centered clusters")
    print("")
    all_iao = []
    for centered_cluster in structure.centered_clusters:
        centered_cluster.print_log()
        print("")
        for ceq_cluster in centered_cluster.centered_equivalent_clusters:
            ceq_cluster.print_log()
            print("")
            ceq_cluster.set_symmetry(find_additional_symmetry = find_additional_symmetry)
            sub_iaos = get_InteractingAOSubspaces_from_cluster(ceq_cluster)
            sub_iaos.print_log()
            print("")
            all_iao += sub_iaos.subspaceslist

    combined_system = CombinedAOSubspaceInteraction(all_iao)
    combined_system.print_log()
    print("")
    
    relationship = OrbitalPropertyRelationship.from_structure_combinedInteraction(
        structure.cell, structure.positions, structure.types, combined_system
    )

    if save_filename:
        relationship.to_file(save_filename)
        print(f"# Orbital relationship is written to output file \"{save_filename}\"")
        print("")
    print(f"Job Done @ "+ time.strftime(r"%Y/%m/%d %H:%M:%S", time.localtime()))