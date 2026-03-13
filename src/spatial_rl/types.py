from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

JSONDict = Dict[str, Any]


@dataclass(slots=True)
class Observation:
    image: np.ndarray
    concept: str
    step_index: int


@dataclass(slots=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: JSONDict


@dataclass(slots=True)
class TrajectoryStep:
    observation: Observation
    action: JSONDict
    action_text: str
    valid_action: bool


@dataclass(slots=True)
class Rollout:
    concept: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    terminal_image: np.ndarray | None = None
    reward_total: float = 0.0
    reward_vis: float = 0.0
    reward_struct: float = 0.0
    terminated_reason: str = ""
    invalid_actions: int = 0


@dataclass(slots=True)
class ActionSample:
    action: JSONDict
    action_text: str
    valid: bool
