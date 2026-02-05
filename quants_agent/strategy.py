from abc import ABC, abstractmethod
from typing import Any
from quants_agent.events import SignalEvent


class Strategy(ABC):
    """
    Strategy is an abstract base class providing an interface for
    all subsequent (inherited) Strategy handling objects.
    """

    @abstractmethod
    def calculate_signals(self, event: Any) -> None:
        """
        Provides the mechanisms to calculate the list of signals.
        """
        raise NotImplementedError
