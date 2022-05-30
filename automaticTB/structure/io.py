from ase import io, Atoms

def read_cif_to_cpt(filename: str) -> tuple:
    struct: Atoms = io.read(filename,format='cif')
    return struct.get_cell(), struct.get_scaled_positions(), struct.get_atomic_numbers()
