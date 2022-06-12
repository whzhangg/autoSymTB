import yaml, json
from ase import io, Atoms

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