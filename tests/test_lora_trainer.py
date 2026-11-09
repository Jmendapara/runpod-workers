"""Unit tests for the LoRA trainer's config generation + progress parsing.

Run: python3 tests/test_lora_trainer.py   (no GPU, no ai-toolkit, no network)
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainers", "lora-trainer"))

from run_log import RunLog, url_secrets  # noqa: E402
from trainer_config import (  # noqa: E402
    arch_env,
    checkpoints_ready,
    pick_error_line,
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


class CheckpointTests(unittest.TestCase):
    def test_saves_at_the_requested_interval_and_keeps_every_file(self):
        job = validate_input(base_input(train={"steps": 1000, "lr": 1e-4, "rank": 16, "save_every": 250}))
        save = build_config(job, "/tmp/w")["config"]["process"][0]["save"]
        self.assertEqual(save["save_every"], 250)
        self.assertGreaterEqual(save["max_step_saves_to_keep"], 1000 // 250)

    def test_an_unusable_interval_means_only_at_the_end(self):
        for bad in ("soon", 0, -5, 5000):
            job = validate_input(base_input(train={"steps": 1000, "lr": 1e-4, "rank": 16, "save_every": bad}))
            self.assertEqual(build_config(job, "/tmp/w")["config"]["process"][0]["save"]["save_every"], 1000)

    def test_finds_step_files_oldest_first_and_skips_the_rest(self):
        names = [
            "rex_000000500.safetensors", "rex_000000250.safetensors", "rex.safetensors", "rex_000001000.safetensors",
            "optimizer.pt", "other_000000250.safetensors", "rex_000000250.safetensors.tmp",
        ]
        self.assertEqual(checkpoints_ready(names, "rex", 1000, set()), [
            (250, [(None, "rex_000000250.safetensors")]),
            (500, [(None, "rex_000000500.safetensors")]),
        ])

    def test_skips_what_is_already_done(self):
        names = ["rex_000000250.safetensors", "rex_000000500.safetensors"]
        self.assertEqual(checkpoints_ready(names, "rex", 1000, {250}), [(500, [(None, "rex_000000500.safetensors")])])

    def test_a_two_stage_step_needs_both_files(self):
        names = ["rex_000000250_high_noise.safetensors", "rex_000000250_low_noise.safetensors", "rex_000000500_high_noise.safetensors"]
        self.assertEqual(checkpoints_ready(names, "rex", 1000, set()), [
            (250, [("high", "rex_000000250_high_noise.safetensors"), ("low", "rex_000000250_low_noise.safetensors")]),
        ])

    def test_the_final_file_still_wins_with_step_files_around(self):
        names = ["rex_000000250.safetensors", "rex_000000500.safetensors", "rex.safetensors"]
        self.assertEqual(pick_output_loras(names, "rex"), [(None, "rex.safetensors")])
        wan = ["rex_000000250_high_noise.safetensors", "rex_000000250_low_noise.safetensors", "rex_high_noise.safetensors", "rex_low_noise.safetensors"]
        self.assertEqual(pick_output_loras(wan, "rex"), [("high", "rex_high_noise.safetensors"), ("low", "rex_low_noise.safetensors")])


class WorkerEnvTests(unittest.TestCase):
    def test_flux_moves_off_cudnn_attention(self):
        self.assertEqual(arch_env("flux"), {"TORCH_CUDNN_SDPA_DEPRIORITIZED": "1"})

    def test_other_architectures_are_unchanged(self):
        for arch in ("flux2_klein_9b", "chroma", "zimage", "qwen_image_2", "krea2", "sdxl", "sd1", "wan22_14b"):
            self.assertEqual(arch_env(arch), {}, arch)

    def test_callers_cannot_change_the_table(self):
        arch_env("flux")["TORCH_CUDNN_SDPA_DEPRIORITIZED"] = "0"
        self.assertEqual(arch_env("flux")["TORCH_CUDNN_SDPA_DEPRIORITIZED"], "1")


class PickErrorLineTests(unittest.TestCase):
    def test_reports_the_exception_not_the_lines_pytorch_prints_after_it(self):
        exception = "torch.AcceleratorError: CUDA error: an illegal memory access was encountered"
        tail = [
            'File "/opt/venv/lib/python3.12/site-packages/torch/autograd/graph.py", line 979, in _engine_run_backward',
            exception,
            "Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.",
            "CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.",
            "For debugging consider passing CUDA_LAUNCH_BLOCKING=1",
        ]
        self.assertEqual(pick_error_line(tail), exception)

    def test_other_failures_still_report_their_last_error_line(self):
        tail = ["loading", "OSError: first", "huggingface_hub.errors.GatedRepoError: 403 Client Error.", "Access to model x is restricted"]
        self.assertEqual(pick_error_line(tail), "huggingface_hub.errors.GatedRepoError: 403 Client Error.")

    def test_a_cudnn_failure_keeps_its_own_line(self):
        line = "RuntimeError: Expected mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)"
        self.assertEqual(pick_error_line([line, "For debugging consider passing CUDA_LAUNCH_BLOCKING=1"]), line)

    def test_nothing_to_report(self):
        self.assertIsNone(pick_error_line(["Running 1 process", "Done"]))
        self.assertIsNone(pick_error_line([]))


class RunLogTests(unittest.TestCase):
    def make(self, **kw):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.uploads = []
        self.now = 1000.0
        upload = kw.pop("upload", lambda path: self.uploads.append(open(path, encoding="utf-8").read()))
        return RunLog(os.path.join(self.dir.name, "run.log"), "users/u/loras/t/logs/run.log", upload, clock=lambda: self.now, **kw)

    def test_hides_secrets(self):
        log = self.make(secrets=["hf_secret", None, "https://x/control?token=abc"])
        log.write("token hf_secret and https://x/control?token=abc here")
        log.upload(force=True)
        self.assertIn("token [redacted] and [redacted] here", self.uploads[0])
        self.assertNotIn("hf_secret", self.uploads[0])
        self.assertNotIn("abc", self.uploads[0])

    def test_hides_a_url_token_when_an_error_quotes_only_the_path(self):
        url = "https://x.test/api/lora-training-control/tr_a?token=abc123"
        log = self.make(secrets=url_secrets(url))
        log.write("Max retries exceeded with url: /api/lora-training-control/tr_a?token=abc123 (Caused by ...)")
        log.write(f"failed to reach {url}")
        log.upload(force=True)
        self.assertNotIn("abc123", self.uploads[0])
        self.assertEqual(url_secrets(None), [])

    def test_uploads_on_a_schedule_and_only_when_there_is_something_new(self):
        log = self.make(every_s=120)
        log.upload()
        self.assertEqual(len(self.uploads), 0)  # nothing written yet
        log.write("a")
        log.upload()
        self.assertEqual(len(self.uploads), 1)  # first upload goes straight away
        log.write("b")
        log.upload()
        self.assertEqual(len(self.uploads), 1)  # not due yet
        log.upload(force=True)
        self.assertEqual(len(self.uploads), 2)
        log.upload(force=True)
        self.assertEqual(len(self.uploads), 2)  # nothing new
        self.now += 121
        log.write("c")
        log.upload()
        self.assertEqual(len(self.uploads), 3)

    def test_a_failed_upload_never_raises_and_is_tried_again(self):
        calls = []

        def flaky(path):
            calls.append(path)
            if len(calls) == 1:
                raise RuntimeError("R2 is down")

        log = self.make(upload=flaky)
        log.write("a")
        log.upload(force=True)
        log.upload(force=True)
        self.assertEqual(len(calls), 2)

    def test_stops_growing_at_the_cap(self):
        log = self.make(max_bytes=100)
        for i in range(50):
            log.write(f"line {i}")
        log.upload(force=True)
        self.assertIn("log truncated", self.uploads[0])
        self.assertLess(len(self.uploads[0]), 200)

    def test_close_removes_the_file_and_later_writes_are_ignored(self):
        log = self.make()
        log.write("a")
        log.close()
        self.assertFalse(os.path.exists(log.path))
        log.write("b")


if __name__ == "__main__":
    unittest.main(verbosity=1)
