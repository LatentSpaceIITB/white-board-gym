import unittest

from spatial_rl.parser import parse_action_text


class ParserTests(unittest.TestCase):
    def test_parses_embedded_json(self):
        raw = 'model output:\n```json\n{"type":"rect","x":10,"y":12,"w":30,"h":40}\n```'
        result = parse_action_text(raw)
        self.assertTrue(result.valid)
        self.assertEqual(result.action["type"], "rect")
        self.assertEqual(result.action["w"], 30.0)

    def test_invalid_returns_finish(self):
        result = parse_action_text("this is not json")
        self.assertFalse(result.valid)
        self.assertEqual(result.action["type"], "finish")

    def test_finish_keyword(self):
        result = parse_action_text("finish")
        self.assertTrue(result.valid)
        self.assertEqual(result.action, {"type": "finish"})


if __name__ == "__main__":
    unittest.main()
