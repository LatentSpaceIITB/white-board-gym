from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from spatial_rl.reward.structural import StructuralScore, compute_structural_score
from spatial_rl.reward.visual import VisualJudge
from spatial_rl.types import JSONDict


@dataclass(slots=True)
class TerminalRewardConfig:
    alpha: float
    beta: float
    width: int
    height: int


class TerminalReward:
    def __init__(self, config: TerminalRewardConfig, visual_judge: VisualJudge):
        self.config = config
        self.visual_judge = visual_judge

    def __call__(
        self, image: np.ndarray, concept: str, actions: Sequence[Mapping[str, object]]
    ) -> tuple[float, JSONDict]:
        r_vis = float(self.visual_judge.score(image, concept))
        struct: StructuralScore = compute_structural_score(
            actions, width=self.config.width, height=self.config.height
        )
        r_struct = struct.score

        total = self.config.alpha * r_vis + self.config.beta * r_struct
        info: JSONDict = {
            "reward_total": float(total),
            "reward_vis": float(r_vis),
            "reward_struct": float(r_struct),
            "has_labels": struct.has_labels,
            "has_arrows": struct.has_arrows,
            "in_bounds": struct.in_bounds,
        }
        return float(total), info
