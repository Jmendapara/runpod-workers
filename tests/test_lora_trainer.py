"""Unit tests for the LoRA trainer's config generation + progress parsing.

Run: python3 tests/test_lora_trainer.py   (no GPU, no ai-toolkit, no network)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainers", "lora-trainer"))

from trainer_config import (  # noqa: E402
    pick_output_loras,
    _sample_every, build_config, caption_for, parse_progress_line, previews_ready, slugify, validate_input,
)


def base_input(**overrides):
    inp = {
        "op": "train",
        "training_id": "tr_abc123XYZ",
        "uid": "user1",
        "name": "My Dog Rex",
        "trigger_word": "rexdog",
        "control_url": "https://qa.example.com/api/lora-training-control/tr_abc123XYZ?token=t",
        "model": {
            "arch": "zimage",
            "name_or_path": "Tongyi-MAI/Z-Image-Turbo",
            "quantize": True,
            "qtype": "qfloat8",
            "quantize_te": True,
            "qtype_te": "qfloat8",
            "low_vram": False,
            "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
        },
        "dataset": {
            "items": [
                {"r2_key": "users/user1/lora-datasets/a.jpg", "kind": "image", "caption": "rexdog sitting on grass"},
                {"r2_key": "users/user1/lora-datasets/b.png", "kind": "image"},
            ],
            "default_caption": "rexdog",
            "resolution": 1024,
            "caption_dropout_rate": 0.05,
        },
        "train": {"steps": 1000, "lr": 1e-4, "rank": 16, "alpha": 16, "timestep_type": "weighted"},
        "sample": {"enabled": True, "prompts": ["rexdog, portrait photo"], "width": 1024, "height": 1024,
                   "sample_steps": 9, "guidance_scale": 1},
        "output": {"prefix": "users/user1/loras/tr_abc123XYZ/", "filename": "my-dog-rex.safetensors"},
    }
    inp.update(overrides)
    return inp


class ValidateInputTests(unittest.TestCase):
    def test_accepts_minimal(self):
        spec = validate_input(base_input())
        self.assertEqual(spec["name"], "my-dog-rex")
        self.assertEqual(spec["train"]["steps"], 1000)
        self.assertEqual(spec["dataset"]["items"][1]["caption"], None)
        self.assertEqual(spec["dataset"]["items"][0]["ext"], ".jpg")

    def test_rejects_bad_steps(self):
        inp = base_input()
        inp["train"]["steps"] = 0
        with self.assertRaises(ValueError):
            validate_input(inp)
        inp["train"]["steps"] = 1.5
        with self.assertRaises(ValueError):
            validate_input(inp)

    def test_rejects_path_traversal(self):
        inp = base_input()
        inp["dataset"]["items"][0]["r2_key"] = "../etc/passwd.jpg"
        with self.assertRaises(ValueError):
            validate_input(inp)
        inp = base_input()
        inp["output"]["prefix"] = "/abs/path/"
        with self.assertRaises(ValueError):
            validate_input(inp)
        inp = base_input()
        inp["output"]["filename"] = "../x.safetensors"
        with self.assertRaises(ValueError):
            validate_input(inp)

    def test_rejects_unknown_extension(self):
        inp = base_input()
        inp["dataset"]["items"][0]["r2_key"] = "users/u/lora-datasets/a.gif"
        with self.assertRaises(ValueError):
            validate_input(inp)

    def test_video_kind_inferred(self):
        inp = base_input()
        inp["dataset"]["items"].append({"r2_key": "users/u/lora-datasets/c.mp4"})
        spec = validate_input(inp)
        self.assertEqual(spec["dataset"]["items"][2]["kind"], "video")

    def test_prompt_cap_and_strip(self):
        inp = base_input()
        inp["sample"]["prompts"] = ["  a ", "", "b", "c", "d", "e", "f"]
        spec = validate_input(inp)
        self.assertEqual(spec["sample"]["prompts"], ["a", "b", "c", "d"])


class OutputLoraTests(unittest.TestCase):
    def test_single_file(self):
        self.assertEqual(pick_output_loras(["job.safetensors", "job_000000500.safetensors", "optimizer.pt"], "job"), [(None, "job.safetensors")])

    def test_two_stage_model_keeps_both(self):
        names = ["job_high_noise.safetensors", "job_low_noise.safetensors", "job_000000500_high_noise.safetensors"]
        self.assertEqual(pick_output_loras(names, "job"), [("high", "job_high_noise.safetensors"), ("low", "job_low_noise.safetensors")])

    def test_step_saves_highest_wins(self):
        self.assertEqual(pick_output_loras(["job_000000500.safetensors", "job_000001000.safetensors"], "job"), [(None, "job_000001000.safetensors")])

    def test_two_stage_step_saves(self):
        names = ["job_000000500_high_noise.safetensors", "job_000000500_low_noise.safetensors", "job_000001000_high_noise.safetensors", "job_000001000_low_noise.safetensors"]
        self.assertEqual(pick_output_loras(names, "job"), [("high", "job_000001000_high_noise.safetensors"), ("low", "job_000001000_low_noise.safetensors")])

    def test_nothing(self):
        self.assertEqual(pick_output_loras(["optimizer.pt"], "job"), [])


class BuildConfigTests(unittest.TestCase):
    def test_image_model_config(self):
        spec = validate_input(base_input())
        cfg = build_config(spec, "/workspace/jobs/tr_abc123XYZ")
        self.assertEqual(cfg["job"], "extension")
        self.assertEqual(cfg["config"]["name"], "my-dog-rex")
        p = cfg["config"]["process"][0]
        self.assertEqual(p["type"], "sd_trainer")
        self.assertEqual(p["training_folder"], "/workspace/jobs/tr_abc123XYZ/output")
        self.assertEqual(p["trigger_word"], "rexdog")
        self.assertEqual(p["network"], {"type": "lora", "linear": 16, "linear_alpha": 16})
        self.assertEqual(p["save"]["save_every"], 1000)
        ds = p["datasets"][0]
        self.assertEqual(ds["folder_path"], "/workspace/jobs/tr_abc123XYZ/dataset")
        self.assertEqual(ds["resolution"], [1024])
        self.assertNotIn("num_frames", ds)
        self.assertEqual(p["train"]["steps"], 1000)
        self.assertEqual(p["train"]["lr"], 1e-4)
        self.assertEqual(p["train"]["noise_scheduler"], "flowmatch")
        self.assertEqual(p["train"]["timestep_type"], "weighted")
        self.assertTrue(p["train"]["skip_first_sample"])
        self.assertFalse(p["train"]["disable_sampling"])
        self.assertEqual(p["model"]["arch"], "zimage")
        self.assertEqual(p["model"]["assistant_lora_path"], base_input()["model"]["assistant_lora_path"])
        self.assertEqual(p["sample"]["prompts"], ["rexdog, portrait photo"])
        self.assertEqual(p["sample"]["sample_every"], 1000)
        self.assertEqual(p["sample"]["guidance_scale"], 1.0)
        self.assertEqual(p["sample"]["sample_steps"], 9)

    def test_video_model_config(self):
        inp = base_input(model={"arch": "wan22_14b", "name_or_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
                                "quantize": True, "qtype": "qfloat8", "low_vram": False,
                                "model_kwargs": {"train_high_noise": True, "train_low_noise": True}})
        inp["dataset"]["items"] = [{"r2_key": "users/u/lora-datasets/c.mp4", "kind": "video", "caption": "rexdog runs"}]
        inp["dataset"].update({"resolution": 512, "num_frames": 33, "fps": 16})
        inp["train"].update({"timestep_type": "linear", "switch_boundary_every": 10, "cache_text_embeddings": True})
        inp["sample"] = {"enabled": False, "prompts": []}
        spec = validate_input(inp)
        cfg = build_config(spec, "/w")
        p = cfg["config"]["process"][0]
        ds = p["datasets"][0]
        self.assertEqual(ds["num_frames"], 33)
        self.assertEqual(ds["fps"], 16)
        self.assertIs(ds["shrink_video_to_frames"], False)
        self.assertEqual(p["train"]["optimizer_params"], {"weight_decay": 1e-4})
        self.assertTrue(p["train"]["disable_sampling"])
        self.assertEqual(p["train"]["switch_boundary_every"], 10)
        self.assertTrue(p["train"]["cache_text_embeddings"])
        self.assertEqual(p["model"]["model_kwargs"], {"train_high_noise": True, "train_low_noise": True})

    def test_sample_every(self):
        self.assertEqual(_sample_every({"every": 500}, 1000), 500)
        self.assertEqual(_sample_every({}, 1000), 1000)
        self.assertEqual(_sample_every({"every": 5000}, 1000), 1000)
        self.assertEqual(_sample_every({"every": "x"}, 1000), 1000)

    def test_previews_ready_skips_final_done_and_other_files(self):
        names = [
            "1700000000__000000500_0.jpg",
            "1700000100__000001000_0.jpg",
            "1700000000__000000500_1.png",
            "notes.txt",
            "1700000000__000000250_0.jpg",
        ]
        ready = previews_ready(names, 1000, {"1700000000__000000250_0.jpg"})
        self.assertEqual(ready, [(500, "1700000000__000000500_0.jpg"), (500, "1700000000__000000500_1.png")])

    def test_latent_cache_defaults_on_and_can_be_turned_off(self):
        cfg = build_config(validate_input(base_input()), "/w")
        self.assertTrue(cfg["config"]["process"][0]["datasets"][0]["cache_latents_to_disk"])
        inp = base_input()
        inp["dataset"]["cache_latents"] = False
        cfg = build_config(validate_input(inp), "/w")
        self.assertFalse(cfg["config"]["process"][0]["datasets"][0]["cache_latents_to_disk"])

    def test_sd_models_use_ddpm(self):
        inp = base_input(model={"arch": "sdxl", "name_or_path": "stabilityai/stable-diffusion-xl-base-1.0", "quantize": False})
        inp["sample"]["guidance_scale"] = 6
        spec = validate_input(inp)
        p = build_config(spec, "/w")["config"]["process"][0]
        self.assertEqual(p["train"]["noise_scheduler"], "ddpm")
        self.assertEqual(p["sample"]["sampler"], "ddpm")

    def test_network_ignore_if_contains(self):
        inp = base_input(network={"ignore_if_contains": ["adaln_proj"]})
        p = build_config(validate_input(inp), "/w")["config"]["process"][0]
        self.assertEqual(p["network"]["network_kwargs"], {"ignore_if_contains": ["adaln_proj"]})


class CaptionTests(unittest.TestCase):
    def test_trigger_prepended_once(self):
        self.assertEqual(caption_for({"caption": "a dog on grass"}, "rexdog", None), "rexdog, a dog on grass")
        self.assertEqual(caption_for({"caption": "rexdog on grass"}, "rexdog", None), "rexdog on grass")
        self.assertEqual(caption_for({"caption": "photo of [trigger] smiling"}, "rexdog", None), "photo of rexdog smiling")

    def test_defaults(self):
        self.assertEqual(caption_for({}, "rexdog", None), "rexdog")
        self.assertEqual(caption_for({}, None, "a style"), "a style")
        self.assertEqual(caption_for({"caption": None}, None, None), "")


class ProgressParseTests(unittest.TestCase):
    def test_tqdm_line(self):
        line = "my-dog-rex:  12%|█▏        | 120/1000 [01:30<11:00,  1.33it/s, lr: 1.0e-04 loss: 3.213e-01]"
        p = parse_progress_line(line, 1000)
        self.assertEqual(p["step"], 120)
        self.assertEqual(p["total"], 1000)
        self.assertEqual(p["elapsed_s"], 90)
        self.assertEqual(p["remaining_s"], 660)
        self.assertAlmostEqual(p["loss"], 0.3213)

    def test_other_totals_ignored(self):
        # a tqdm bar with a different total is not a training step; the caching hint still surfaces
        self.assertEqual(parse_progress_line("Caching latents: 10/20 [00:01<00:01, 9.0it/s]", 1000), {"phase": "caching"})
        self.assertIsNone(parse_progress_line("epoch 10/20 [x", 1000))

    def test_phase_hints(self):
        self.assertEqual(parse_progress_line("Loading Z-Image model", 1000), {"phase": "loading"})
        self.assertEqual(parse_progress_line("Generating Images: 0%", 1000), {"phase": "sampling"})
        self.assertEqual(parse_progress_line("Saving at step 1000", 1000), {"phase": "saving"})
        self.assertIsNone(parse_progress_line("random noise", 1000))

    def test_slugify(self):
        self.assertEqual(slugify("My Dog Rex!!"), "my-dog-rex")
        self.assertEqual(slugify("   "), "lora")
        self.assertEqual(slugify("a" * 100), "a" * 48)


if __name__ == "__main__":
    unittest.main(verbosity=1)
