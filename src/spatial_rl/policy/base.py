from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from spatial_rl.types import ActionSample, JSONDict, Observation


@dataclass(slots=True)
class PolicyContext:
    width: int
    height: int


class BasePolicy(ABC):
    @abstractmethod
    def sample_action(
        self, observation: Observation, *, temperature: float = 1.0
    ) -> ActionSample:
        raise NotImplementedError

    @abstractmethod
    def logprob_action(self, observation: Observation, action: Mapping[str, object]):
        raise NotImplementedError

    @abstractmethod
    def train_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def clone_frozen(self) -> "BasePolicy":
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> JSONDict:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: JSONDict) -> None:
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError
