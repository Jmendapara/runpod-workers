"""Unit tests for the TRELLIS.2-4B worker's input contract.

Run: python3 tests/test_trellis2.py   (no GPU, no torch, no network)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainers", "trellis2-4b"))

from worker_config import InputError, MAX_FACES, MIN_FACES, output_key, parse_input  # noqa: E402

KEY = "users/u1/uploads/abc-photo.png"


class ParseInput(unittest.TestCase):
    def test_defaults_match_the_official_demo(self):
        spec = parse_input({"image_r2_key": KEY})
        self.assertEqual(spec.pipeline_type, "1024_cascade")
        self.assertEqual(spec.quality, "standard")
        self.assertEqual(spec.texture_size, 2048)
        self.assertEqual(spec.decimation_target, 500_000)
        self.assertEqual(spec.seed, 42)
        self.assertTrue(spec.turntable)
        self.assertIsNone(spec.uid)

    def test_high_quality_is_the_1536_cascade(self):
        spec = parse_input({"image_r2_key": KEY, "quality": "HIGH", "seed": "7", "texture_size": 4096, "uid": "u1"})
        self.assertEqual(spec.pipeline_type, "1536_cascade")
        self.assertEqual(spec.seed, 7)
        self.assertEqual(spec.texture_size, 4096)
        self.assertEqual(spec.uid, "u1")

    def test_refuses_bad_values(self):
        bad = [
            None,
            {},
            {"image_r2_key": ""},
            {"image_r2_key": "../etc/passwd"},
            {"image_r2_key": "/abs/key.png"},
            {"image_r2_key": KEY, "quality": "ultra"},
            {"image_r2_key": KEY, "seed": -1},
            {"image_r2_key": KEY, "seed": 2**31},
            {"image_r2_key": KEY, "seed": 1.5},
            {"image_r2_key": KEY, "seed": True},
            {"image_r2_key": KEY, "texture_size": 3000},
            {"image_r2_key": KEY, "decimation_target": MIN_FACES - 1},
            {"image_r2_key": KEY, "decimation_target": MAX_FACES + 1},
            {"image_r2_key": KEY, "turntable": "yes"},
            {"image_r2_key": KEY, "uid": "a/b"},
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(InputError):
                parse_input(raw)

    def test_output_keys_follow_the_comfyui_workers(self):
        self.assertEqual(output_key("x.glb", "job1", "u1"), "users/u1/generations/x.glb")
        self.assertEqual(output_key("x.glb", "job1", None), "job1/x.glb")


if __name__ == "__main__":
    unittest.main()
