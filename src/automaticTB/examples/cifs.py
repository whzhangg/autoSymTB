import os

filenames = {
    "Si": "Si_ICSD51688.cif",
    "PbTe": "PbTe_ICSD38295.cif",
    "Mg2Sn": "Mg2Sn_mp-2343.cif",
    "Perovskite": "perovskite.cif"
}

current_folder = os.path.split(__file__)[0]
resource_folder = os.path.join(current_folder, "..", "..", "..", "resources")

def get_cif_filename_for(structure_name: str) -> str:
    """
    return stored cif file name
    """
    if structure_name not in filenames:
        print("Only these structure are stored")
        print(list(filenames.keys()))
        SystemExit
    
    return os.path.join(resource_folder, filenames[structure_name])