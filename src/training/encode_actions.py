"""Encode/decode WhiteboardGym actions as compact token strings for LM training.

Raw STROKE points are fit to a cubic Bézier (4 control points = 8 integers)
so every action becomes a fixed-format string that Qwen2-VL can predict.

Usage:
    from src.training.encode_actions import encode_action, decode_action

    tok = encode_action(action_dict)   # e.g. "S149,121,152,118,155,114,160,109,2"
    back = decode_action(tok)          # round-trips to dict
"""

import numpy as np
from scipy.interpolate import splprep, splev

ACTION_TYPE_TO_ID = {
    "STROKE": 0,
    "TEXT": 1,
    "ERASE": 2,
    "SETCOLOR": 3,
    "SETWIDTH": 4,
    "WAIT": 5,
    "AUDIOSYNC": 6,
    "FINISH": 7,
}

ID_TO_ACTION_TYPE = {v: k for k, v in ACTION_TYPE_TO_ID.items()}


def _fit_bezier(pts: list) -> list:
    """Fit cubic spline to raw points, sample at 4 fixed t-values -> 4 control points."""
    # Deduplicate consecutive identical points (splprep fails on zero-length segments)
    deduped = [pts[0]] if pts else [[0, 0]]
    for p in pts[1:]:
        if p != deduped[-1]:
            deduped.append(p)

    if len(deduped) < 2:
        p = deduped[0]
        return [list(p), list(p), list(p), list(p)]

    arr = np.array(deduped, dtype=float)
    k = min(3, len(arr) - 1)
    tck, _ = splprep([arr[:, 0], arr[:, 1]], s=0, k=k)
    t_vals = [0.0, 0.333, 0.667, 1.0]
    xs, ys = splev(t_vals, tck)
    return [[int(round(x)), int(round(y))] for x, y in zip(xs, ys)]


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert '#rrggbb' to (r, g, b) ints."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert (r, g, b) to '#rrggbb'."""
    return f"#{r:02x}{g:02x}{b:02x}"


def encode_action(action: dict) -> str:
    """Convert an action dict to a compact token string."""
    t = action["type"]

    if t == "STROKE":
        ctrl = _fit_bezier(action["points"])
        w = int(round(action.get("width", 1)))
        coords = ",".join(f"{p[0]},{p[1]}" for p in ctrl)
        return f"S{coords},{w}"

    elif t == "TEXT":
        x = int(round(action["x"]))
        y = int(round(action["y"]))
        fs = int(round(action.get("font_size_norm", action.get("font_size", 12))))
        content = action.get("content", "")
        return f"T{x},{y},{fs},{content}"

    elif t == "ERASE":
        cx = int(round(action["x"]))
        cy = int(round(action["y"]))
        w = int(round(action.get("width", 10)))
        h = int(round(action.get("height", 10)))
        return f"E{cx},{cy},{w},{h}"

    elif t == "SETCOLOR":
        r, g, b = _hex_to_rgb(action["color"])
        return f"C{r},{g},{b}"

    elif t == "SETWIDTH":
        w = int(round(action["width"]))
        return f"L{w}"

    elif t == "WAIT":
        bins = min(300, max(0, int(round(action["duration"] / 0.1))))
        return f"W{bins}"

    elif t == "AUDIOSYNC":
        idx = int(action.get("segment_index", 0))
        return f"A{idx}"

    elif t == "FINISH":
        return "F"

    else:
        raise ValueError(f"Unknown action type: {t}")


def decode_action(token: str) -> dict:
    """Convert a compact token string back to an action dict."""
    prefix = token[0]
    body = token[1:]

    if prefix == "S":
        parts = body.split(",")
        if len(parts) != 9:
            raise ValueError(f"STROKE token needs 9 values, got {len(parts)}: {token}")
        nums = [int(x) for x in parts]
        points = [[nums[i], nums[i + 1]] for i in range(0, 8, 2)]
        return {"type": "STROKE", "points": points, "width": nums[8]}

    elif prefix == "T":
        # Format: T<x>,<y>,<fs>,<content>  — content may contain commas
        parts = body.split(",", 3)
        return {
            "type": "TEXT",
            "x": int(parts[0]),
            "y": int(parts[1]),
            "font_size_norm": int(parts[2]),
            "content": parts[3] if len(parts) > 3 else "",
        }

    elif prefix == "E":
        parts = body.split(",")
        return {
            "type": "ERASE",
            "x": int(parts[0]),
            "y": int(parts[1]),
            "width": int(parts[2]),
            "height": int(parts[3]),
        }

    elif prefix == "C":
        parts = body.split(",")
        return {
            "type": "SETCOLOR",
            "color": _rgb_to_hex(int(parts[0]), int(parts[1]), int(parts[2])),
        }

    elif prefix == "L":
        return {"type": "SETWIDTH", "width": int(body)}

    elif prefix == "W":
        bins = int(body)
        return {"type": "WAIT", "duration": round(bins * 0.1, 1)}

    elif prefix == "A":
        return {"type": "AUDIOSYNC", "segment_index": int(body)}

    elif prefix == "F":
        return {"type": "FINISH"}

    else:
        raise ValueError(f"Unknown token prefix: {prefix!r} in {token!r}")
