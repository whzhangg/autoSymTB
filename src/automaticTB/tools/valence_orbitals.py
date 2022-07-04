valence_dictionary = {
    'H': [0], 'He': [0], 
    'Li': [0], 'Be': [0], 'B': [0, 1], 'C': [0, 1], 'N': [0, 1], 'O': [0, 1], 'F': [0, 1], 'Ne': [0, 1], 
    'Na': [0], 'Mg': [0], 'Al': [0, 1], 'Si': [0, 1], 'P': [0, 1], 'S': [0, 1], 'Cl': [0, 1], 'Ar': [0, 1], 
    'K': [0], 'Ca': [0], 'Sc': [2], 'Ti': [2], 'V': [2], 'Cr': [2], 'Mn': [2], 'Fe': [2], 'Co': [2], 
    'Ni': [2], 'Cu': [2], 'Zn': [2], 'Ga': [0, 1], 'Ge': [0, 1], 'As': [0, 1], 'Se': [0, 1], 'Br': [0, 1], 
    'Kr': [0, 1], 'Rb': [0], 'Sr': [0], 'Y': [2], 'Zr': [2], 'Nb': [2], 'Mo': [2], 'Tc': [2], 'Ru': [2], 
    'Rh': [2], 'Pd': [2], 'Ag': [2], 'Cd': [2], 'In': [0, 1], 'Sn': [0, 1], 'Sb': [0, 1], 'Te': [0, 1], 
    'I': [0, 1], 'Xe': [0, 1], 'Cs': [0], 'Ba': [0], 'Hf': [2], 'Ta': [2], 'W': [2], 'Re': [2], 'Os': [2], 
    'Ir': [2], 'Pt': [2], 'Au': [2], 'Hg': [2], 'Tl': [0, 1], 'Pb': [0, 1], 'Bi': [0, 1], 'Po': [0, 1], 
    'Rn': [0, 1]
}

"""
def get_orbital_info():
    from DFTtools.pseudoinfo import pp_info
    result = {}
    for key, property in pp_info.items():
        try:
            suitable_orbitals = property["wannierproj"].split(";")
            suitable_l = [int(so.split("=")[1]) for so in suitable_orbitals]
            result[key] = suitable_l
        except:
            pass
    return result
"""