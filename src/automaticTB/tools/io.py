import typing
import json
import yaml


_print_lm = {
    (0, 0): "s",
    (1,-1): "py",
    (1, 0): "pz",
    (1, 1): "px",
    (2,-2): "dxy",
    (2,-1): "dyz",
    (2, 0): "dz2",
    (2, 1): "dxz",
    (2, 2): "dx2-y2"
}

def get_orbital_symbol_from_lm(l: int, m: int) -> str:
    if (l,m) not in _print_lm:
        raise "l,m not supported"
    return _print_lm[(l,m)]


def parse_orbital(n: int, l: int, m: int) -> str:
    if (l,m) not in _print_lm:
        return f"n={n};l={l};m={m}"
    else:
        formatted = _print_lm[(l,m)]
        return f"{n}{formatted}"


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


def format_lines_into_two_columns(
    lines1: typing.List[str], lines2: typing.List[str], 
    spacing: typing.Optional[int] = None
) -> typing.List[str]:
    """
    this function take two list of strings and combine them together into two column lines
    """
    if spacing is None:
        longest = max([ len(l) for l in lines1 ] + [ len(l) for l in lines2 ]) * 1.6
        longest = max(30, longest)
        spacing = int(longest)

    formatstring = "{:<"+str(spacing)+"s}{:<"+str(spacing)+"s}"
    if len(lines1) > len(lines2):
        lines2 += [""] * ( len(lines1) - len(lines2) )
    else:
        lines1 += [""] * ( len(lines2) - len(lines1) )

    result = []
    for l1, l2 in zip(lines1, lines2):
        result.append(formatstring.format(l1, l2))
    
    return result