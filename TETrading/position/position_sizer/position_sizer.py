from abc import ABCMeta, abstractmethod
from typing import Dict


class PositionSizer:

    """
    Base class for classes with position sizing functionality.
    """

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def position_size_metric_str(self):
        ...
    
    @abstractmethod
    def __call__(self, dt, weights) -> Dict:
        raise NotImplementedError('Should implement call()')
