from abc import ABCMeta, abstractmethod


class PositionSizer:

    """
    Base class for classes with position sizing functionality.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, dt, weights):
        raise NotImplementedError('Should implement call()')
