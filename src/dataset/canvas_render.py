"""Render action sequences to canvas images (C_t states for RL training).

C_t is the canvas state BEFORE action t is applied. For a sequence of N actions,
render_all() returns N+1 images: C_0 (blank) ... C_N (final canvas after FINISH).

Usage:
    from src.dataset.canvas_render import CanvasRenderer
    import json

    with open("data/processed/video-2.json") as f:
        episode = json.load(f)

    renderer = CanvasRenderer()
    frames = renderer.render_all(episode["actions"])
    # frames[i] = PIL Image of canvas BEFORE episode["actions"][i]
    frames[3].save("c3_before_text.png")
"""

import os
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a system font at the given pixel size, falling back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except (IOError, OSError):
                continue
    return ImageFont.load_default()


class CanvasRenderer:
    """Renders a WhiteboardGym action sequence to a series of PIL Images."""

    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height

    def blank(self) -> Image.Image:
        """Return a fresh white canvas."""
        return Image.new("RGB", (self.width, self.height), "white")

    def apply_action(self, canvas: Image.Image, action: dict) -> Image.Image:
        """Return a new canvas with the action rendered onto it.

        Non-visual actions (WAIT, SETCOLOR, SETWIDTH, AUDIOSYNC, FINISH)
        return a copy of the canvas unchanged.
        """
        canvas = canvas.copy()
        draw = ImageDraw.Draw(canvas)
        t = action["type"]

        if t == "STROKE":
            pts = [tuple(p) for p in action["points"]]
            color = action.get("color", "#000000")
            width = max(1, int(action.get("width", 1)))
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=width)
            elif len(pts) == 1:
                x, y = pts[0]
                draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)

        elif t == "TEXT":
            font_size = action.get("font_size_norm", 12)
            font = _load_font(font_size)
            x, y = action["x"], action["y"]
            content = action.get("content", "")
            # x, y are the normalized center — anchor to middle-left
            try:
                draw.text((x, y), content, fill="#000000", font=font, anchor="lm")
            except TypeError:
                # Older PIL without anchor support
                draw.text((x, y), content, fill="#000000", font=font)

        elif t == "ERASE":
            cx, cy = action["x"], action["y"]
            w2 = action.get("width", 10) // 2
            h2 = action.get("height", 10) // 2
            draw.rectangle([cx - w2, cy - h2, cx + w2, cy + h2], fill="white")

        # WAIT, SETCOLOR, SETWIDTH, AUDIOSYNC, FINISH → no pixel change

        return canvas

    def render_all(self, actions: List[dict]) -> List[Image.Image]:
        """Render C_t for every step in the action sequence.

        Returns a list of len(actions)+1 images:
          frames[i] = canvas state BEFORE actions[i] is applied
          frames[-1] = final canvas after all actions

        This is the full set of (C_t, τ_t) observation states needed for training.
        """
        canvas = self.blank()
        frames = [canvas.copy()]  # C_0 = blank canvas before action 0
        for action in actions:
            canvas = self.apply_action(canvas, action)
            frames.append(canvas.copy())
        return frames

    def render_at(self, actions: List[dict], step: int) -> Image.Image:
        """Render the canvas state just before actions[step] is applied."""
        canvas = self.blank()
        for action in actions[:step]:
            canvas = self.apply_action(canvas, action)
        return canvas

    def save_frames(
        self,
        actions: List[dict],
        output_dir: str,
        steps: Optional[List[int]] = None,
    ) -> List[str]:
        """Render and save canvas frames as PNG files.

        Args:
            actions:    Full action sequence from episode JSON.
            output_dir: Directory to write PNGs into.
            steps:      Specific step indices to save. If None, saves all.

        Returns:
            List of file paths written.
        """
        os.makedirs(output_dir, exist_ok=True)
        frames = self.render_all(actions)
        indices = steps if steps is not None else range(len(frames))

        paths = []
        for i in indices:
            if 0 <= i < len(frames):
                path = os.path.join(output_dir, f"frame_{i:04d}.png")
                frames[i].save(path)
                paths.append(path)
        return paths
