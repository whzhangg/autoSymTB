import typing, abc
from .interaction_pairs import AOPair, AOPairWithValue


__all__ = ["AOEquationBase"]


class AOEquationBase(abc.ABC):
    """
    the equation base object, inherited by InteractionEquation and CombinedEquation
    """
    @property
    @abc.abstractmethod
    def free_AOpairs(self) -> typing.List[AOPair]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def all_AOpairs(self) -> typing.List[AOPair]:
        raise NotImplementedError

    @abc.abstractmethod
    def solve_interactions_to_InteractionPairs(self, values: typing.List[float]) \
    -> typing.List[AOPairWithValue]:
        raise NotImplementedError