"""Compute global bounding box and normalize coordinates to 256×256."""

from typing import List, Tuple
from .parse_excalidraw import ExcalidrawElement


class Normalizer:
    """Scale arbitrary canvas coordinates to [0, 255]."""

    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.dx = max(max_x - min_x, 1.0)
        self.dy = max(max_y - min_y, 1.0)

    def norm_x(self, x: float) -> int:
        return max(0, min(255, int((x - self.min_x) / self.dx * 255)))

    def norm_y(self, y: float) -> int:
        return max(0, min(255, int((y - self.min_y) / self.dy * 255)))

    def norm_point(self, x: float, y: float) -> List[int]:
        return [self.norm_x(x), self.norm_y(y)]


def build_normalizer(elements: List[ExcalidrawElement]) -> Normalizer:
    """Compute global bounding box across all elements and return a Normalizer."""
    all_x: List[float] = []
    all_y: List[float] = []

    for elem in elements:
        if elem.type == "freedraw":
            for pt in elem.points:
                all_x.append(elem.x + pt[0])
                all_y.append(elem.y + pt[1])
        elif elem.type == "text":
            all_x.append(elem.x)
            all_x.append(elem.x + elem.width)
            all_y.append(elem.y)
            all_y.append(elem.y + elem.height)

    if not all_x:
        return Normalizer(0, 255, 0, 255)

    return Normalizer(
        min_x=min(all_x),
        max_x=max(all_x),
        min_y=min(all_y),
        max_y=max(all_y),
    )
