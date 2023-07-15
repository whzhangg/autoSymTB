import typing, time
import numpy as np

from automaticTB.tools import read_cif_to_cpt, format_lines_into_two_columns, LinearEquation
from automaticTB.solve.structure import Structure
from automaticTB.solve.interaction import (
    AOSubspace,
    InteractionSpace,
    get_InteractingAOSubspaces_from_cluster,
)
from automaticTB.interface import OrbitalPropertyRelationship

def solve_interaction(
    structure: typing.Union[str, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]],
    orbitals_dict: typing.Dict[str, str],
    rcut: typing.Optional[float] = None,
    standardize: bool = True,
    find_additional_symmetry: bool = False,
    degenerate_atomic_orbitals: bool = True,
    save_filename: str = None, 
    return_type: typing.Literal["OrbitalPropertyRelationship", "CombinedInteraction"] \
        = "CombinedInteraction",
    experimental: bool = True
) -> OrbitalPropertyRelationship:
    """
    this function wraps the process of finding the unique interaction 
    parameter. It return `None` but will print the output log 
    information in a similar fashion of traditional calculation program.
    It will write a file which can be read to perform extra 
    operation/property extraction. 
    """

    print(f"Program start @ " + time.strftime(r"%Y/%m/%d %H:%M:%S", time.localtime()))
    print("")
    if type(structure) == str:
        c, p, t = read_cif_to_cpt(structure)
        print(f"# Reading crystal structure from {structure}")
    else:
        print(f"# Using given cell, positions and types")
        c, p, t = structure
    structure: Structure = Structure(c, p, t, orbitals_dict, rcut, standardize)

    print("")
    structure.print_log()
    print("")

    gathered_solution = []
    gathered_clusters = []
    for eq_index, eq_clusters in structure.equivalent_clusters.items():

        print(f"# Solving interaction for non-eq cluster with p-index {eq_index+1}")
        print("-"*75)
        selected = eq_clusters[0]
        selected.print_log()
        print("")
        blocks = []
        for ceq_cluster in selected.centered_equivalent_clusters:
            ceq_cluster.set_symmetry(
                find_additional_symmetry = find_additional_symmetry,
                degenerate_atomic_orbital=degenerate_atomic_orbitals
            )
            print(f"## site symmetry: {ceq_cluster.sitesymmetrygroup.groupname}")
            subspace_pairs = get_InteractingAOSubspaces_from_cluster(ceq_cluster)
            print_log_for_InteractingAOSubspaces(subspace_pairs)
            print("")
            print("## Solutions")
            total_number_pairs = 0
            for l_sub, r_sub in subspace_pairs:
                subspace = InteractionSpace.from_AOSubspace(l_sub, r_sub, verbose=True)
                total_number_pairs += len(subspace.free_indices)
                blocks.append(subspace)
            print("")
            print(f'## Total number of free interactions = {total_number_pairs}')
            print("")
        combined_interaction = InteractionSpace.from_Blocks(blocks)
        gathered_solution.append(combined_interaction)
        gathered_clusters.append(eq_clusters)
        print("-"*75)

    print("")
    print(f"# Use it to generate other positions")
    print("")

    if experimental:
        combined_equivalent = InteractionSpace.from_solvedInteraction_and_symmetry_new(
                gathered_solution, gathered_clusters, structure.cartesian_rotations, verbose=True
        )
    else:
        combined_equivalent = InteractionSpace.from_solvedInteraction_and_symmetry(
                gathered_solution, gathered_clusters, structure.cartesian_rotations, verbose=True
        )

    print("")
    print("# Interaction Treatment Finished ! @ ", 
            time.strftime(r"%Y/%m/%d %H:%M:%S", time.localtime()))
    print("")
    relationship = OrbitalPropertyRelationship.from_structure_combinedInteraction(
        structure.cell, structure.positions, structure.types, combined_equivalent
    )

    print("# The final free interaction pairs are:")
    relationship.print_free_pairs()
    print("")
    if save_filename:
        relationship.to_file(save_filename)
        print(f"# Orbital relationship is written to output file \"{save_filename}\"")
        print("")
    print(f"Job Done @ "+ time.strftime(r"%Y/%m/%d %H:%M:%S", time.localtime()))

    if return_type == "OrbitalPropertyRelationship":
        return relationship
    else:
        return combined_equivalent


def print_log_for_InteractingAOSubspaces(                                                          
    subspaceslist: typing.List[typing.Tuple[AOSubspace, AOSubspace]]
) -> None:
    print(f"## Orbital interaction subspace")
    l_str = {0:"s", 1:"p", 2:"d", 3:"f"}
    l_orbitals: typing.Dict[str, typing.List[str]] = {}
    r_orbitals: typing.Dict[str, typing.List[str]] = {}
        
    for subspace in subspaceslist:
        tmp = subspace[0].aos[0]
        l_orb = f"{tmp.chemical_symbol:>2s}({tmp.n:>2d}{l_str[tmp.l]})"
        l_orbitals[l_orb] = [str(nlc.name) for nlc in subspace[0].namedlcs]

        tmp = subspace[1].aos[0]
        r_orb = f"{tmp.chemical_symbol:>2s}({tmp.n:>2d}{l_str[tmp.l]})"
        r_orbitals[r_orb] = [str(nlc.name) for nlc in subspace[1].namedlcs]

    max_length = 25
    for vs in l_orbitals.values():
        max_length = max(max_length, max(len(v) for v in vs))
    for vs in r_orbitals.values():
        max_length = max(max_length, max(len(v) for v in vs))
    max_length = int(max_length * 1.5)

    l_lines = ["Left ---"]
    for k, vs in l_orbitals.items():
        l_lines.append("")
        l_lines.append(k)
        for v in vs:
            l_lines.append(v)

    r_lines = ["Right --"]
    for k, vs in r_orbitals.items():
        r_lines.append("")
        r_lines.append(k)
        for v in vs:
            r_lines.append(v)

    formatted = format_lines_into_two_columns(l_lines, r_lines)
    for f in formatted:
        print(f"  {f}")
