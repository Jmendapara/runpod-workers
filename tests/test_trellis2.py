"""Unit tests for the TRELLIS.2-4B worker's input contract.

Run: python3 tests/test_trellis2.py   (no GPU, no torch, no network)
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainers", "trellis2-4b"))

from worker_config import InputError, MAX_FACES, MIN_FACES, output_key, parse_input  # noqa: E402

try:
    import numpy as np
    import views
except ImportError:  # numpy is optional on a laptop; the view tests skip without it
    np = None

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


@unittest.skipIf(np is None, "numpy not installed")
class Views(unittest.TestCase):
    def picture(self, flip=False, color=(0.8, 0.2, 0.1)):
        # An L-shaped object (tall bar + foot to one side) on black, like the pipeline's cut-out.
        img = np.zeros((200, 160, 3), np.float32)
        img[20:180, 40:70] = color
        if flip:
            img[150:180, 70:140] = (0.1, 0.3, 0.9)
        else:
            img[150:180, 0:40] = (0.1, 0.3, 0.9)
        return img

    def test_the_matching_view_wins_over_its_mirror_and_others(self):
        target = views.target_from_cutout(self.picture())
        same = views.normalize_view(self.picture(), self.picture().max(axis=2) > 0.03)
        mirror = views.normalize_view(self.picture(flip=True), self.picture(flip=True).max(axis=2) > 0.03)
        blob = np.zeros((100, 100, 3), np.float32); blob[30:70, 30:70] = 0.5
        other = views.normalize_view(blob, blob.max(axis=2) > 0)
        yaws = [0.0, 1.0, 2.0, 3.0]
        yaw, score = views.best_yaw(target, [mirror, other, same, None], yaws)
        self.assertEqual(yaw, 2.0)
        self.assertGreater(score, 0.9)

    def test_scale_and_position_do_not_matter(self):
        small = np.zeros((400, 400, 3), np.float32)
        small[100:180, 300:315] = (0.8, 0.2, 0.1)
        small[165:180, 280:300] = (0.1, 0.3, 0.9)
        big = np.zeros((300, 300, 3), np.float32)
        big[20:280, 100:150] = (0.8, 0.2, 0.1)
        big[230:280, 33:100] = (0.1, 0.3, 0.9)
        a = views.target_from_cutout(small)
        b = views.normalize_view(big, big.max(axis=2) > 0.03)
        self.assertGreater(views.score_view(a, b), 0.6)

    def test_dark_parts_inside_the_object_still_count(self):
        img = np.zeros((100, 100, 3), np.float32)
        img[10:90, 10:90] = 0.7
        img[40:60, 40:60] = 0.0  # a black detail in the middle of the object
        _, mask = views.target_from_cutout(img)
        self.assertTrue(mask[views.MATCH_SIZE // 2, views.MATCH_SIZE // 2])

    def test_empty_views_never_win(self):
        target = views.target_from_cutout(self.picture())
        self.assertEqual(views.score_view(target, None), -1.0)
        self.assertIsNone(views.normalize_view(np.zeros((10, 10, 3)), np.zeros((10, 10), bool)))

    def test_viewer_orbit_maps_trellis_axes_to_gltf(self):
        # TRELLIS camera at +X (yaw 90°) is at +X in the GLB: model-viewer theta 90°.
        self.assertAlmostEqual(views.viewer_orbit(math.radians(90), 0.0)[0], 90.0)
        # At -Y (yaw 180°, TRELLIS) the GLB camera is at +Z: theta 0.
        self.assertAlmostEqual(views.viewer_orbit(math.radians(180), 0.0)[0] % 360, 0.0)
        theta, phi = views.viewer_orbit(math.radians(119), 0.30)
        self.assertAlmostEqual(theta, 61.0, places=3)
        self.assertAlmostEqual(phi, 90 - math.degrees(0.30), places=3)


if __name__ == "__main__":
    unittest.main()
