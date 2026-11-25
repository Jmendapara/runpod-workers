"""Unit tests for the TRELLIS.2-4B worker's input contract.

Run: python3 tests/test_trellis2.py   (no GPU, no torch, no network)
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainers", "trellis2-4b"))

from worker_config import InputError, MAX_FACES, MAX_IMAGES, MIN_FACES, output_key, parse_input  # noqa: E402
import multi_image  # noqa: E402

try:
    import numpy as np
    import picture
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

    def test_one_picture_is_a_list_of_one(self):
        spec = parse_input({"image_r2_key": KEY})
        self.assertEqual(spec.image_r2_keys, (KEY,))
        self.assertEqual(spec.image_r2_key, KEY)
        self.assertEqual(parse_input({"image_r2_keys": [KEY]}).image_r2_keys, (KEY,))

    def test_several_pictures_keep_their_order_and_the_first_is_the_main_one(self):
        keys = [f"users/u1/uploads/{side}.png" for side in ("front", "back", "left", "right")]
        spec = parse_input({"image_r2_keys": keys, "image_r2_key": "ignored.png"})
        self.assertEqual(spec.image_r2_keys, tuple(keys))
        self.assertEqual(spec.image_r2_key, keys[0])
        self.assertEqual(spec.together_steps, 4)
        self.assertEqual(parse_input({"image_r2_keys": keys, "together_steps": 0}).together_steps, 0)
        self.assertIsNone(spec.sampler_steps)
        self.assertEqual(parse_input({"image_r2_keys": keys, "sampler_steps": 24}).sampler_steps, 24)

    def test_refuses_bad_picture_lists(self):
        bad = [
            {"image_r2_keys": []},
            {"image_r2_keys": KEY},
            {"image_r2_keys": [KEY, ""]},
            {"image_r2_keys": [KEY, "../x.png"]},
            {"image_r2_keys": [KEY, 5]},
            {"image_r2_keys": [KEY, KEY]},
            {"image_r2_keys": [f"k{i}.png" for i in range(MAX_IMAGES + 1)]},
            {"image_r2_keys": [KEY, "b.png"], "together_steps": -1},
            {"image_r2_keys": [KEY, "b.png"], "together_steps": 51},
            {"image_r2_keys": [KEY, "b.png"], "together_steps": "some"},
            {"image_r2_keys": [KEY, "b.png"], "sampler_steps": 3},
            {"image_r2_keys": [KEY, "b.png"], "sampler_steps": 51},
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(InputError):
                parse_input(raw)

    def test_output_keys_follow_the_comfyui_workers(self):
        self.assertEqual(output_key("x.glb", "job1", "u1"), "users/u1/generations/x.glb")
        self.assertEqual(output_key("x.glb", "job1", None), "job1/x.glb")


class Cond:
    """A stand-in for a conditioning tensor: a batch of labelled pictures."""
    def __init__(self, items):
        self.items = list(items)
        self.shape = (len(self.items),)

    def __getitem__(self, s):
        return Cond(self.items[s])


class MultiImage(unittest.TestCase):
    T = (1.0, 0.9, 0.7, 0.4, 0.2)  # a five-step sampler: every stage starts at t = 1

    def sampler(self, count, together_steps, stage_steps=(5,)):
        calls = []

        def original(self, model, x_t, t, cond=None, **kwargs):
            calls.append((cond.items, kwargs))
            return float(sum(cond.items))  # the "prediction" for a picture is its label

        return multi_image.make_inference(original, count, together_steps, list(stage_steps)), calls

    def test_together_is_the_mean_of_one_prediction_per_picture(self):
        infer, calls = self.sampler(3, together_steps=5)
        pred = infer(None, "model", "x", 1.0, Cond([3.0, 6.0, 9.0]), concat_cond="shape")
        self.assertAlmostEqual(pred, 6.0)
        self.assertEqual([c[0] for c in calls], [[3.0], [6.0], [9.0]])
        self.assertTrue(all(c[1] == {"concat_cond": "shape"} for c in calls))

    def test_the_negative_conditioning_passes_through_once(self):
        infer, calls = self.sampler(3, together_steps=5)
        self.assertEqual(infer(None, "model", "x", 0.8, Cond([0.0])), 0.0)
        self.assertEqual(len(calls), 1)

    def test_in_turn_takes_one_picture_per_step_and_restarts_each_stage(self):
        infer, _ = self.sampler(2, together_steps=0)
        cond, neg = Cond([1.0, 2.0]), Cond([0.0])
        seen = []
        for _stage in range(2):
            for t in self.T:
                seen.append(infer(None, "model", "x", t, cond))
                infer(None, "model", "x", t, neg)  # guidance's negative call must not advance the turn
        self.assertEqual(seen, [1.0, 2.0, 1.0, 2.0, 1.0] * 2)

    def test_turns_first_then_the_last_steps_together(self):
        infer, calls = self.sampler(2, together_steps=2)
        cond = Cond([1.0, 3.0])
        seen = [infer(None, "model", "x", t, cond) for t in self.T]
        self.assertEqual(seen, [1.0, 3.0, 1.0, 2.0, 2.0])
        self.assertEqual(len(calls), 3 + 2 * 2)

    def test_each_stage_counts_its_own_steps(self):
        # A three-step stage, then a five-step one: the last two steps of each are taken together.
        infer, _ = self.sampler(2, together_steps=2, stage_steps=(3, 5))
        cond = Cond([1.0, 3.0])
        first = [infer(None, "model", "x", t, cond) for t in (1.0, 0.6, 0.2)]
        second = [infer(None, "model", "x", t, cond) for t in self.T]
        self.assertEqual(first, [1.0, 2.0, 2.0])
        self.assertEqual(second, [1.0, 3.0, 1.0, 2.0, 2.0])

    def test_more_together_steps_than_the_sampler_has_means_every_step(self):
        infer, _ = self.sampler(2, together_steps=50)
        self.assertEqual([infer(None, "model", "x", t, Cond([1.0, 3.0])) for t in self.T], [2.0] * 5)

    def test_the_sampler_class_is_restored_even_after_an_error(self):
        class Sampler:
            def _inference_model(self, model, x_t, t, cond=None, **kwargs):
                return "original"

        before = Sampler._inference_model
        with self.assertRaises(RuntimeError):
            with multi_image.several_pictures(Sampler, 2, 4, [12]):
                self.assertIsNot(Sampler._inference_model, before)
                raise RuntimeError("boom")
        self.assertIs(Sampler._inference_model, before)


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


@unittest.skipIf(np is None, "numpy not installed")
class Picture(unittest.TestCase):
    def rgba(self, background_alpha, subject_alpha=255):
        img = np.zeros((100, 100, 4), np.uint8)
        img[..., :3] = 255
        img[..., 3] = background_alpha
        img[30:70, 40:60, 3] = subject_alpha
        return img

    def test_an_invisible_residue_of_the_old_background_is_dropped(self):
        out = picture.clean_alpha(self.rgba(background_alpha=12))
        self.assertIsNotNone(out)
        self.assertEqual(int(out[0, 0, 3]), 0)
        self.assertEqual(int(out[50, 50, 3]), 255)

    def test_soft_edges_of_the_subject_are_kept(self):
        img = self.rgba(background_alpha=0)
        img[30, 40:60, 3] = 128
        self.assertEqual(int(picture.clean_alpha(img)[30, 45, 3]), 128)

    def test_an_alpha_that_cuts_nothing_out_is_ignored(self):
        self.assertIsNone(picture.clean_alpha(self.rgba(background_alpha=255)))
        # a 1 px transparent border around a 1024 px picture is not a cut-out either
        img = np.full((1024, 1024, 4), 255, np.uint8)
        img[0, :, 3] = img[-1, :, 3] = img[:, 0, 3] = img[:, -1, 3] = 0
        self.assertIsNone(picture.clean_alpha(img))


if __name__ == "__main__":
    unittest.main()
