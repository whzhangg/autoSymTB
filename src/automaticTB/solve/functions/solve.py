import typing, time
import numpy as np

from automaticTB.tools import read_cif_to_cpt, format_lines_into_two_columns
from automaticTB.solve.structure import Structure
from automaticTB.solve.interaction import (
    InteractingAOSubspace,
    get_InteractingAOSubspaces_from_cluster,
    BlockInteractions,
    CombinedEquivalentInteractingAO
)
from automaticTB.interface import OrbitalPropertyRelationship


def solve_interaction(
    structure: typing.Union[str, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]],
    orbitals_dict: typing.Dict[str, str],
    rcut: typing.Optional[float] = None,
    standardize: bool = True,
    find_additional_symmetry: bool = False,
    save_filename: str = None
) -> OrbitalPropertyRelationship:
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
    structure: Structure = Structure(c, p, t, orbitals_dict, rcut, standardize)

    print("")
    structure.print_log()
    print("")

    combined_equivalents = []
    for eq_index, eq_clusters in structure.equivalent_clusters.items():

        print(f"# Solving interaction for non-eq cluster with p-index {eq_index+1}")
        print("")
        selected = eq_clusters[0]
        selected.print_log()
        print("")
        aosubs = []
        for ceq_cluster in selected.centered_equivalent_clusters:
            ceq_cluster.set_symmetry(find_additional_symmetry = find_additional_symmetry)
            subspaces = get_InteractingAOSubspaces_from_cluster(ceq_cluster)
            print_log_for_InteractingAOSubspaces(subspaces)
            print("")
            aosubs += subspaces
        combined_interaction = BlockInteractions(aosubs)
        combined_interaction.print_log()
        print("")
        print(f"# Use it to generate other positions")
        print("")
        generated = eq_clusters[1:]
        combined_equivalent = CombinedEquivalentInteractingAO(
            selected, combined_interaction,
            generated, structure.cartesian_rotations
        )
        combined_equivalent.print_log()
        print("")
        combined_equivalents.append(combined_equivalent)

    combined_system = BlockInteractions(combined_equivalents)
    #combined_system.print_log()
    #print("")

    relationship = OrbitalPropertyRelationship.from_structure_combinedInteraction(
        structure.cell, structure.positions, structure.types, combined_system
    )

    print("# The final free interaction pairs are:")
    relationship.print_free_pairs()
    print("")
    if save_filename:
        relationship.to_file(save_filename)
        print(f"# Orbital relationship is written to output file \"{save_filename}\"")
        print("")
    print(f"Job Done @ "+ time.strftime(r"%Y/%m/%d %H:%M:%S", time.localtime()))

    return relationship

def print_log_for_InteractingAOSubspaces(
    subspaceslist: typing.List["InteractingAOSubspace"]
) -> None:
    print(f"## Orbital interaction subspace")
    l_str = {0:"s", 1:"p", 2:"d", 3:"f"}
    l_orbitals: typing.Dict[str, typing.List[str]] = {}
    r_orbitals: typing.Dict[str, typing.List[str]] = {}
        
    for subspace in subspaceslist:
        tmp = subspace.l_subspace.aos[0]
        l_orb = f"{tmp.chemical_symbol:>2s}({tmp.n:>2d}{l_str[tmp.l]})"
        l_orbitals[l_orb] = [str(nlc.name) for nlc in subspace.l_subspace.namedlcs]

        tmp = subspace.r_subspace.aos[0]
        r_orb = f"{tmp.chemical_symbol:>2s}({tmp.n:>2d}{l_str[tmp.l]})"
        r_orbitals[r_orb] = [str(nlc.name) for nlc in subspace.r_subspace.namedlcs]

    max_length = -1
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
    print("")
    print(f"## Detailed orbital interactions for each subspace")
    for subspace in subspaceslist:
        tmp = subspace.l_subspace.aos[0]
        l_orb = f"{tmp.chemical_symbol:>2s}({tmp.n:>2d}{l_str[tmp.l]})"

        tmp = subspace.r_subspace.aos[0]
        r_orb = f"{tmp.chemical_symbol:>2s}({tmp.n:>2d}{l_str[tmp.l]})"
        print(
            f"### {l_orb} -> {r_orb} (free/total) orb. interactions:" + 
            f" ({len(subspace.free_AOpairs)}/{len(subspace.all_AOpairs)})"
        )
        if subspace.free_AOpairs:
            for i, f in enumerate(subspace.free_AOpairs):
                print(f"  {i+1:>3d} " + str(f))

