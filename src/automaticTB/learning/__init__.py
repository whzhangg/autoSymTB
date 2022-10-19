"""
This module should be able to use the power of e3nn package to map for orbital interactions. 
for demonstration purpose, it should provide a function that take interaction pairs with their 
respective values as training inputs and return a function that provide rotation, permutation 
invariant mapping of the interaction value
"""

from .interaction_mapper import OrbitalNetwork