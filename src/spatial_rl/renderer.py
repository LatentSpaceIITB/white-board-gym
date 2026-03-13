from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spatial_rl.types import JSONDict


@dataclass(slots=True)
class RendererConfig:
    width: int
    height: int
    background_color: str = "#ffffff"


class CanvasRenderer:
    def __init__(self, config: RendererConfig):
        self.config = config
        self._font = ImageFont.load_default()

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    def blank_canvas(self) -> np.ndarray:
        img = Image.new(
            "RGB", (self.width, self.height), color=self.config.background_color
        )
        return np.array(img, dtype=np.uint8)

    def render_action(
        self, canvas: np.ndarray, action: Mapping[str, object]
    ) -> np.ndarray:
        action_type = action.get("type")
        if action_type == "finish":
            return canvas.copy()

        image = Image.fromarray(canvas.copy(), mode="RGB")
        draw = ImageDraw.Draw(image)

        if action_type == "freedraw":
            points = [tuple(float(v) for v in p) for p in action["pts"]]
            width = max(1, int(round(float(action["w"]))))
            draw.line(points, fill=str(action["col"]), width=width, joint="curve")

        elif action_type == "rect":
            x = float(action["x"])
            y = float(action["y"])
            w = float(action["w"])
            h = float(action["h"])
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=str(action["cs"]),
                fill=str(action["cf"]),
                width=2,
            )

        elif action_type == "ellipse":
            x = float(action["x"])
            y = float(action["y"])
            rx = float(action["rx"])
            ry = float(action["ry"])
            draw.ellipse(
                [x - rx, y - ry, x + rx, y + ry],
                outline=str(action["cs"]),
                fill=str(action["cf"]),
                width=2,
            )

        elif action_type == "text":
            x = float(action["x"])
            y = float(action["y"])
            text = str(action["s"])
            draw.text(
                (x, y), text, fill=str(action.get("col", "#111827")), font=self._font
            )

        elif action_type == "arrow":
            x1 = float(action["x1"])
            y1 = float(action["y1"])
            x2 = float(action["x2"])
            y2 = float(action["y2"])
            color = str(action.get("col", "#111827"))
            self._draw_arrow(draw, x1, y1, x2, y2, color)
            label = str(action.get("l", "")).strip()
            if label:
                mx = (x1 + x2) / 2.0
                my = (y1 + y2) / 2.0
                draw.text((mx + 3, my + 3), label, fill=color, font=self._font)

        return np.array(image, dtype=np.uint8)

    def render_program(self, actions: list[JSONDict]) -> np.ndarray:
        canvas = self.blank_canvas()
        for action in actions:
            canvas = self.render_action(canvas, action)
        return canvas

    @staticmethod
    def _draw_arrow(
        draw: ImageDraw.ImageDraw,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
    ) -> None:
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

        angle = math.atan2(y2 - y1, x2 - x1)
        head_len = 10.0
        left = (
            x2 - head_len * math.cos(angle - math.pi / 6),
            y2 - head_len * math.sin(angle - math.pi / 6),
        )
        right = (
            x2 - head_len * math.cos(angle + math.pi / 6),
            y2 - head_len * math.sin(angle + math.pi / 6),
        )
        draw.polygon([(x2, y2), left, right], fill=color)
