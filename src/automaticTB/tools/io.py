import yaml, json, typing
from ase import io, Atoms

__all__ = [
    "atom_from_cpt", "write_json", "read_json",
    "write_yaml", "read_yaml", "read_cif_to_cpt",
    "format_lines_into_two_columns"
]


def atom_from_cpt(lattice, positions, types) -> Atoms:
    result = Atoms(cell = lattice, scaled_positions = positions)
    if type(types[0]) == str:
        result.set_chemical_symbols(types)
    else:
        result.set_atomic_numbers(types)
    return result
    

def write_json(dictionary, filename):
    f=open(filename,'w')
    json.dump(dictionary,f,indent="    ")
    f.close()


def read_json(filename):
    f=open(filename,'r')
    data=json.load(f)
    f.close()
    return data


def write_yaml(data, filename: str):
    # it is able to save python objects
    yaml_str = yaml.dump(data)
    with open(filename, 'w') as f:
        f.write(yaml_str)


def read_yaml(filename: str):
    with open(filename, 'r') as f:
        groupdata = yaml.load(f, Loader=yaml.Loader)
    return groupdata


def read_cif_to_cpt(filename: str) -> tuple:
    struct: Atoms = io.read(filename,format='cif')
    return struct.get_cell(), struct.get_scaled_positions(), struct.get_atomic_numbers()


def format_lines_into_two_columns(
    lines1: typing.List[str], lines2: typing.List[str], 
    spacing: typing.Optional[int] = None
) -> str:
    """
    this function take two list of strings and combine them together into two column lines
    """
    if spacing is None:
        longest = max([ len(l) for l in lines1 ] + [ len(l) for l in lines2 ]) * 1.6
        spacing = int(longest)

    formatstring = "{:<"+str(spacing)+"s}{:<"+str(spacing)+"s}"
    if len(lines1) > len(lines2):
        lines2 += [""] * ( len(lines1) - len(lines2) )
    else:
        lines1 += [""] * ( len(lines2) - len(lines1) )

    result = []
    for l1, l2 in zip(lines1, lines2):
        result.append(formatstring.format(l1, l2))
    
    return "\n".join(result)