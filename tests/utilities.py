import numpy as np

from automaticTB.solve import interaction

def find_interaction_values(referencefilename: str, pairs: interaction.AOPair) -> np.ndarray:
    """find the interactions"""
    interactions = {}
    with open(referencefilename, 'r') as f:
        while aline := f.readline():
            name, value = aline.split(" : ")
            interactions[name] = complex(value)
    
    retrived_values = []
    for pair in pairs:
        entry = str(pair)
        retrived_values.append(interactions[entry])
        
    return np.array(retrived_values)

