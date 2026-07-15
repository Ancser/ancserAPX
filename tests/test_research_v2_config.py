import json
import tempfile
import unittest
from pathlib import Path

from research_v2.config import ResearchConfig, load_config


class ResearchConfigTests(unittest.TestCase):
    def test_default_is_deterministic(self):
        a = ResearchConfig()
        b = ResearchConfig()
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(len(a.fingerprint()), 64)

    def test_json_override_and_unknown_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.json"
            p.write_text(json.dumps({"portfolio": {"top_n": 15}}), encoding="utf-8")
            self.assertEqual(load_config(p).portfolio.top_n, 15)
            p.write_text(json.dumps({"live": {"enabled": True}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(p)


if __name__ == "__main__":
    unittest.main()
