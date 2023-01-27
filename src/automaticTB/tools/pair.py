import typing
import collections


Pair = collections.namedtuple("Pair", "left right")
PairwithValue = collections.namedtuple("PairwithValue", "left right value")


def tensor_dot(list_left, list_right) -> typing.List[Pair]:
    """make a list of items from flattened tensor dot results"""
    r = []
    for item1 in list_left:
        for item2 in list_right:
            r.append(Pair(item1, item2))
    return r