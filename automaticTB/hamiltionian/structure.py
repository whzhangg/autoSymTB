# we for each atom in the primitive cell, we first need to obtain its interaction, NN_interaction, as well as the position
# of the atom in the center, which is not stored in the NN-Interaction
# then, furthermore, we need to know the translation of the atoms.

# to constructure a tight-binding model, we start by creating the list of atomic orbitals, from each NNinteraction, we 
# can fill one row in the interaction: the interaction of the center atoms to all its neighbors, all other information
# is not necessary. 