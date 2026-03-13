from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from spatial_rl.types import JSONDict

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
ACTION_TYPES = {"freedraw", "rect", "ellipse", "text", "arrow", "finish"}


class ActionValidationError(ValueError):
    pass


def normalize_color(value: Any, *, default: str = "#1f2933") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ActionValidationError("color must be a string")
    text = value.strip()
    if not HEX_COLOR_RE.match(text):
        raise ActionValidationError(f"invalid hex color: {value!r}")
    return text.lower()


def _to_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ActionValidationError(f"field '{field_name}' must be numeric") from exc


def _to_nonempty_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ActionValidationError(f"field '{field_name}' must be text")
    text = value.strip()
    if not text:
        raise ActionValidationError(f"field '{field_name}' cannot be empty")
    return text


def _to_points(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ActionValidationError("freedraw 'pts' must contain at least 2 points")

    pts: list[list[float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ActionValidationError(f"point {index} must be [x, y]")
        x = _to_float(point[0], field_name=f"pts[{index}][0]")
        y = _to_float(point[1], field_name=f"pts[{index}][1]")
        pts.append([x, y])
    return pts


def normalize_action(raw_action: Mapping[str, Any]) -> JSONDict:
    if not isinstance(raw_action, Mapping):
        raise ActionValidationError("action must be a JSON object")

    action_type = raw_action.get("type")
    if not isinstance(action_type, str):
        raise ActionValidationError("action 'type' must be a string")

    action_type = action_type.strip().lower()
    if action_type not in ACTION_TYPES:
        raise ActionValidationError(f"unknown action type: {action_type!r}")

    if action_type == "finish":
        return {"type": "finish"}

    if action_type == "freedraw":
        pts = _to_points(raw_action.get("pts"))
        stroke_w = _to_float(raw_action.get("w", 2.0), field_name="w")
        if stroke_w <= 0:
            raise ActionValidationError("freedraw 'w' must be > 0")
        return {
            "type": "freedraw",
            "pts": pts,
            "col": normalize_color(raw_action.get("col")),
            "w": stroke_w,
        }

    if action_type == "rect":
        width = _to_float(raw_action.get("w"), field_name="w")
        height = _to_float(raw_action.get("h"), field_name="h")
        if width <= 0 or height <= 0:
            raise ActionValidationError("rect 'w' and 'h' must be > 0")
        return {
            "type": "rect",
            "x": _to_float(raw_action.get("x"), field_name="x"),
            "y": _to_float(raw_action.get("y"), field_name="y"),
            "w": width,
            "h": height,
            "cs": normalize_color(raw_action.get("cs"), default="#1f2933"),
            "cf": normalize_color(raw_action.get("cf"), default="#dbeafe"),
        }

    if action_type == "ellipse":
        rx = _to_float(raw_action.get("rx"), field_name="rx")
        ry = _to_float(raw_action.get("ry"), field_name="ry")
        if rx <= 0 or ry <= 0:
            raise ActionValidationError("ellipse 'rx' and 'ry' must be > 0")
        return {
            "type": "ellipse",
            "x": _to_float(raw_action.get("x"), field_name="x"),
            "y": _to_float(raw_action.get("y"), field_name="y"),
            "rx": rx,
            "ry": ry,
            "cs": normalize_color(raw_action.get("cs"), default="#1f2933"),
            "cf": normalize_color(raw_action.get("cf"), default="#dcfce7"),
        }

    if action_type == "text":
        font_size = _to_float(raw_action.get("f", 12), field_name="f")
        if font_size <= 0:
            raise ActionValidationError("text 'f' must be > 0")
        return {
            "type": "text",
            "s": _to_nonempty_text(raw_action.get("s"), field_name="s"),
            "x": _to_float(raw_action.get("x"), field_name="x"),
            "y": _to_float(raw_action.get("y"), field_name="y"),
            "f": font_size,
            "col": normalize_color(raw_action.get("col"), default="#111827"),
        }

    if action_type == "arrow":
        label = raw_action.get("l", "")
        if label is None:
            label = ""
        if not isinstance(label, str):
            raise ActionValidationError("arrow label 'l' must be text")
        return {
            "type": "arrow",
            "x1": _to_float(raw_action.get("x1"), field_name="x1"),
            "y1": _to_float(raw_action.get("y1"), field_name="y1"),
            "x2": _to_float(raw_action.get("x2"), field_name="x2"),
            "y2": _to_float(raw_action.get("y2"), field_name="y2"),
            "l": label.strip(),
            "col": normalize_color(raw_action.get("col"), default="#111827"),
        }

    raise ActionValidationError(f"unsupported action type: {action_type!r}")


def action_coordinates(action: Mapping[str, Any]) -> Iterable[tuple[float, float]]:
    action_type = action.get("type")

    if action_type == "finish":
        return []

    if action_type == "freedraw":
        return [(float(p[0]), float(p[1])) for p in action["pts"]]

    if action_type == "rect":
        x = float(action["x"])
        y = float(action["y"])
        w = float(action["w"])
        h = float(action["h"])
        return [(x, y), (x + w, y + h)]

    if action_type == "ellipse":
        x = float(action["x"])
        y = float(action["y"])
        rx = float(action["rx"])
        ry = float(action["ry"])
        return [(x - rx, y - ry), (x + rx, y + ry)]

    if action_type == "text":
        return [(float(action["x"]), float(action["y"]))]

    if action_type == "arrow":
        return [
            (float(action["x1"]), float(action["y1"])),
            (float(action["x2"]), float(action["y2"])),
        ]

    return []


def action_in_bounds(action: Mapping[str, Any], *, width: int, height: int) -> bool:
    for x, y in action_coordinates(action):
        if x < 0 or x > width:
            return False
        if y < 0 or y > height:
            return False
    return True
