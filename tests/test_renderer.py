import hashlib
import unittest

from spatial_rl.renderer import CanvasRenderer, RendererConfig


class RendererTests(unittest.TestCase):
    def test_renderer_is_deterministic(self):
        renderer = CanvasRenderer(
            RendererConfig(width=128, height=128, background_color="#ffffff")
        )
        actions = [
            {
                "type": "rect",
                "x": 12,
                "y": 10,
                "w": 40,
                "h": 30,
                "cs": "#111111",
                "cf": "#dbeafe",
            },
            {
                "type": "arrow",
                "x1": 20,
                "y1": 20,
                "x2": 90,
                "y2": 70,
                "l": "flow",
                "col": "#111111",
            },
            {"type": "text", "s": "label", "x": 50, "y": 80, "f": 12, "col": "#111111"},
        ]

        image_a = renderer.render_program(actions)
        image_b = renderer.render_program(actions)

        hash_a = hashlib.sha256(image_a.tobytes()).hexdigest()
        hash_b = hashlib.sha256(image_b.tobytes()).hexdigest()
        self.assertEqual(hash_a, hash_b)


if __name__ == "__main__":
    unittest.main()
