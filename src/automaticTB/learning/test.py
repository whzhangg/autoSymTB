from automaticTB.learning.orbital_interaction_adopter import (
    get_onehot_element_encoding, 
    OrbitalIrrepsEncoder,
    Orbital,
    InteractingOrbitals
)
import numpy as np


def test_orbital_encoder():
    orb_encoder = OrbitalIrrepsEncoder([(1,0),(2,0),(2,1)])
    assert orb_encoder.irreps_str == "1x0e + 1x0e + 1x1o"
    assert np.allclose(orb_encoder.get_nlm_feature((2,0,0)), np.array([0, 1, 0, 0, 0]))

def test_element_encode():
    elements = {14, 32, 26}
    assert np.allclose(get_onehot_element_encoding(14, elements), np.array([1,0,0]))
    assert np.allclose(get_onehot_element_encoding(26, elements), np.array([0,1,0]))
    assert np.allclose(get_onehot_element_encoding(32, elements), np.array([0,0,1]))


elements = {14, 32, 26}
orb_encoder = OrbitalIrrepsEncoder([(1,0),(2,0),(2,1)])

node1 = {
    "ele": 14,
    "nlm": (1, 0, 0), # 1s
    "pos": np.array([-1, 0, 0])
}
node2 = {
    "ele": 32,
    "nlm": (2, 1, 1), # 1s
    "pos": np.array([2, -1, 1])
}

interaction = InteractingOrbitals(
    Orbital(
        node1["pos"], 
        get_onehot_element_encoding(node1["ele"], elements),
        orb_encoder.get_nlm_feature(node1["nlm"]),
        orb_encoder.irreps_str
    ),
    Orbital(
        node2["pos"], 
        get_onehot_element_encoding(node2["ele"], elements),
        orb_encoder.get_nlm_feature(node2["nlm"]),
        orb_encoder.irreps_str
    ),
    1.5
)

interaction2 = InteractingOrbitals(
    Orbital(
        node1["pos"], 
        get_onehot_element_encoding(node2["ele"], elements),
        orb_encoder.get_nlm_feature(node2["nlm"]),
        orb_encoder.irreps_str
    ),
    Orbital(
        node2["pos"], 
        get_onehot_element_encoding(node1["ele"], elements),
        orb_encoder.get_nlm_feature(node1["nlm"]),
        orb_encoder.irreps_str
    ),
    None
)

from automaticTB.learning.interaction_mapper import (
    get_nearest_neighbor_orbital_mapper, train_model
)

f = get_nearest_neighbor_orbital_mapper(len(elements), orb_encoder.irreps_str)

original_data = interaction.geometry_data
rotated_data = interaction.get_randomlyrotated_pyg_data()

f = train_model(f, [original_data])

print(f(original_data))
print(f(rotated_data))
print(f(interaction2.geometry_data))