from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from spatial_rl.schema import action_in_bounds


@dataclass(slots=True)
class StructuralScore:
    score: float
    has_labels: bool
    has_arrows: bool
    in_bounds: bool


def compute_structural_score(
    actions: Sequence[Mapping[str, object]], *, width: int, height: int
) -> StructuralScore:
    has_labels = any(action.get("type") == "text" for action in actions)
    has_arrows = any(action.get("type") == "arrow" for action in actions)

    bounds_ok = True
    for action in actions:
        if not action_in_bounds(action, width=width, height=height):
            bounds_ok = False
            break

    score = (float(has_labels) + float(has_arrows) + float(bounds_ok)) / 3.0
    return StructuralScore(
        score=score, has_labels=has_labels, has_arrows=has_arrows, in_bounds=bounds_ok
    )
