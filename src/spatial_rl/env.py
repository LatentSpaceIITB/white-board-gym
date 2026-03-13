from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from spatial_rl.renderer import CanvasRenderer, RendererConfig
from spatial_rl.schema import ActionValidationError, normalize_action
from spatial_rl.types import JSONDict, Observation, StepResult

RewardFn = Callable[[np.ndarray, str, list[JSONDict]], tuple[float, JSONDict]]


@dataclass(slots=True)
class EnvConfig:
    width: int
    height: int
    tmax: int
    background_color: str = "#ffffff"


class SpatialDrawingEnv:
    def __init__(self, config: EnvConfig, reward_fn: RewardFn):
        self.config = config
        self._renderer = CanvasRenderer(
            RendererConfig(
                width=config.width,
                height=config.height,
                background_color=config.background_color,
            )
        )
        self._reward_fn = reward_fn

        self._canvas = self._renderer.blank_canvas()
        self._concept = ""
        self._actions: list[JSONDict] = []
        self._step_index = 0
        self._done = False
        self._invalid_actions = 0

    @property
    def actions(self) -> list[JSONDict]:
        return list(self._actions)

    def reset(self, concept: str) -> Observation:
        self._canvas = self._renderer.blank_canvas()
        self._concept = concept
        self._actions = []
        self._step_index = 0
        self._done = False
        self._invalid_actions = 0
        return Observation(
            image=self._canvas.copy(),
            concept=self._concept,
            step_index=self._step_index,
        )

    def step(self, raw_action: JSONDict) -> StepResult:
        if self._done:
            raise RuntimeError("environment step called after episode completion")

        valid = True
        invalid_reason = None
        try:
            action = normalize_action(raw_action)
        except ActionValidationError as exc:
            valid = False
            invalid_reason = str(exc)
            action = {"type": "finish"}
            self._invalid_actions += 1

        self._actions.append(action)
        if action["type"] != "finish":
            self._canvas = self._renderer.render_action(self._canvas, action)

        self._step_index += 1

        terminated = action["type"] == "finish"
        truncated = self._step_index >= self.config.tmax and not terminated
        self._done = terminated or truncated

        reward = 0.0
        info: JSONDict = {
            "valid_action": valid,
            "invalid_reason": invalid_reason,
            "invalid_actions": self._invalid_actions,
        }
        if self._done:
            reward, reward_info = self._reward_fn(
                self._canvas, self._concept, self._actions
            )
            termination_reason = "finish" if terminated else "tmax"
            info.update(reward_info)
            info["termination_reason"] = termination_reason

        observation = Observation(
            image=self._canvas.copy(),
            concept=self._concept,
            step_index=self._step_index,
        )
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
