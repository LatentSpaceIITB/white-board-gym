import unittest

from spatial_rl.reward.structural import compute_structural_score


class StructuralRewardTests(unittest.TestCase):
    def test_perfect_structural_score(self):
        actions = [
            {"type": "text", "s": "part", "x": 10, "y": 10, "f": 12, "col": "#111111"},
            {
                "type": "arrow",
                "x1": 10,
                "y1": 10,
                "x2": 40,
                "y2": 40,
                "l": "flow",
                "col": "#111111",
            },
        ]
        score = compute_structural_score(actions, width=128, height=128)
        self.assertTrue(score.has_labels)
        self.assertTrue(score.has_arrows)
        self.assertTrue(score.in_bounds)
        self.assertAlmostEqual(score.score, 1.0)

    def test_out_of_bounds_penalty(self):
        actions = [
            {"type": "text", "s": "part", "x": 10, "y": 10, "f": 12, "col": "#111111"},
            {
                "type": "arrow",
                "x1": -5,
                "y1": 10,
                "x2": 40,
                "y2": 40,
                "l": "flow",
                "col": "#111111",
            },
        ]
        score = compute_structural_score(actions, width=128, height=128)
        self.assertFalse(score.in_bounds)
        self.assertAlmostEqual(score.score, 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
