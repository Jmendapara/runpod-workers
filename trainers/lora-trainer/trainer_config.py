"""Builds the ai-toolkit job config (YAML) from the OpenSourceGen job input.

Pure functions — no GPU, no network — so they are unit-testable (tests/test_lora_trainer.py).
The server (opensourcegen shared/loraTraining.js) decides WHAT to train (model arch, name_or_path,
quantization, steps, lr, rank, ...); this module only shapes it into ai-toolkit's config schema
(config/examples/*.yaml + toolkit/config_modules.py).
"""
from __future__ import annotations

import os
import re
from typing import Any

# Input validation -----------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
SAFE_NAME_RE = re.compile(r"[^a-z0-9_-]+")

MAX_ITEMS = 200
MAX_STEPS = 10_000
MAX_PROMPTS = 4


def slugify(name: str, fallback: str = "lora") -> str:
    s = SAFE_NAME_RE.sub("-", (name or "").strip().lower()).strip("-_")
    s = re.sub(r"-{2,}", "-", s)
    return (s or fallback)[:48]


def _require(obj: dict, key: str, typ, what: str):
    if key not in obj:
        raise ValueError(f"{what}.{key} is required")
    val = obj[key]
    if typ is float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{what}.{key} must be a number")
        return float(val)
    if typ is int:
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{what}.{key} must be an integer")
        return val
    if not isinstance(val, typ):
        raise ValueError(f"{what}.{key} must be {typ.__name__}")
    return val


def validate_input(inp: dict) -> dict:
    """Validate + normalize the job input. Raises ValueError with a user-safe message."""
    if not isinstance(inp, dict):
        raise ValueError("input must be an object")
    if inp.get("op", "train") != "train":
        raise ValueError(f"unsupported op '{inp.get('op')}'")

    training_id = _require(inp, "training_id", str, "input")
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", training_id):
        raise ValueError("input.training_id is malformed")
    uid = inp.get("uid")
    if uid is not None and (not isinstance(uid, str) or "/" in uid or not uid):
        raise ValueError("input.uid is malformed")

    model = _require(inp, "model", dict, "input")
    arch = _require(model, "arch", str, "model")
    name_or_path = _require(model, "name_or_path", str, "model")
    if not arch or not name_or_path:
        raise ValueError("model.arch and model.name_or_path are required")

    dataset = _require(inp, "dataset", dict, "input")
    items = _require(dataset, "items", list, "dataset")
    if not items:
        raise ValueError("dataset.items is empty")
    if len(items) > MAX_ITEMS:
        raise ValueError(f"dataset.items has more than {MAX_ITEMS} entries")
    norm_items = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"dataset.items[{i}] must be an object")
        key = it.get("r2_key")
        if not isinstance(key, str) or not key or key.startswith("/") or ".." in key:
            raise ValueError(f"dataset.items[{i}].r2_key is malformed")
        ext = os.path.splitext(key)[1].lower()
        kind = it.get("kind") or ("video" if ext in VIDEO_EXTS else "image")
        if kind not in ("image", "video"):
            raise ValueError(f"dataset.items[{i}].kind must be image or video")
        if kind == "image" and ext not in IMAGE_EXTS:
            raise ValueError(f"dataset.items[{i}] has an unsupported image extension '{ext}'")
        if kind == "video" and ext not in VIDEO_EXTS:
            raise ValueError(f"dataset.items[{i}] has an unsupported video extension '{ext}'")
        caption = it.get("caption")
        if caption is not None and not isinstance(caption, str):
            raise ValueError(f"dataset.items[{i}].caption must be a string")
        norm_items.append({"r2_key": key, "kind": kind, "ext": ext, "caption": (caption or "").strip() or None})

    train = _require(inp, "train", dict, "input")
    steps = _require(train, "steps", int, "train")
    if steps < 1 or steps > MAX_STEPS:
        raise ValueError(f"train.steps must be 1..{MAX_STEPS}")
    lr = _require(train, "lr", float, "train")
    if not (0 < lr <= 0.01):
        raise ValueError("train.lr is out of range")
    rank = _require(train, "rank", int, "train")
    if rank < 1 or rank > 256:
        raise ValueError("train.rank must be 1..256")

    sample = inp.get("sample") or {}
    if not isinstance(sample, dict):
        raise ValueError("input.sample must be an object")
    prompts = sample.get("prompts") or []
    if not isinstance(prompts, list) or any(not isinstance(p, str) for p in prompts):
        raise ValueError("sample.prompts must be a list of strings")
    prompts = [p.strip() for p in prompts if p and p.strip()][:MAX_PROMPTS]

    output = _require(inp, "output", dict, "input")
    prefix = _require(output, "prefix", str, "output")
    if not prefix or prefix.startswith("/") or ".." in prefix:
        raise ValueError("output.prefix is malformed")
    filename = output.get("filename") or f"{slugify(inp.get('name') or training_id)}.safetensors"
    if "/" in filename or not filename.endswith(".safetensors"):
        raise ValueError("output.filename must be a bare *.safetensors name")

    control_url = inp.get("control_url")
    if control_url is not None and (not isinstance(control_url, str) or not control_url.startswith("http")):
        raise ValueError("input.control_url must be an http(s) URL")

    hf_token = inp.get("hf_token")
    if hf_token is not None and not isinstance(hf_token, str):
        raise ValueError("input.hf_token must be a string")

    return {
        "training_id": training_id,
        "uid": uid,
        "name": slugify(inp.get("name") or training_id),
        "trigger_word": (inp.get("trigger_word") or "").strip() or None,
        "model": model,
        "dataset": {**dataset, "items": norm_items},
        "train": {**train, "steps": steps, "lr": lr, "rank": rank},
        "network": inp.get("network") or {},
        "sample": {**sample, "prompts": prompts},
        "output": {"prefix": prefix.rstrip("/") + "/", "filename": filename},
        "control_url": control_url,
        "hf_token": hf_token or None,
        "bucket_profile": inp.get("bucket_profile") or None,
    }


# Config generation -----------------------------------------------------------------------------

def _clean(d: dict) -> dict:
    """Drop None values so ai-toolkit falls back to its own defaults."""
    return {k: v for k, v in d.items() if v is not None}


def build_config(job: dict, workdir: str) -> dict:
    """ai-toolkit `job: extension` config for one LoRA training (see config/examples/)."""
    model = job["model"]
    train = job["train"]
    ds = job["dataset"]
    sample = job["sample"]
    network = job["network"]
    name = job["name"]
    steps = train["steps"]
    has_video = any(it["kind"] == "video" for it in ds["items"])
    num_frames = int(ds.get("num_frames") or 1) if has_video else 1

    dataset_cfg = _clean({
        "folder_path": os.path.join(workdir, "dataset"),
        "caption_ext": "txt",
        "default_caption": ds.get("default_caption") or job.get("trigger_word") or name,
        "caption_dropout_rate": float(ds.get("caption_dropout_rate", 0.05)),
        "shuffle_tokens": False,
        # Default on (VAE off the GPU during training). A model that reads raw frames from the batch on
        # every step (Wan 2.2 14B i2v: batch.tensor) needs it off — cached latents leave that tensor empty.
        "cache_latents_to_disk": ds.get("cache_latents") is not False,
        "resolution": [int(ds.get("resolution") or 1024)],
        # Video models look for video files when num_frames > 1 (toolkit/config_modules.py).
        "num_frames": num_frames if num_frames > 1 else None,
        "fps": int(ds["fps"]) if has_video and ds.get("fps") else None,
        "do_i2v": bool(ds["do_i2v"]) if has_video and ds.get("do_i2v") is not None else None,
        "do_audio": bool(ds["do_audio"]) if has_video and ds.get("do_audio") is not None else None,
        "shrink_video_to_frames": True if has_video else None,
    })

    sampling_enabled = bool(sample.get("enabled", True)) and bool(sample.get("prompts"))
    sample_cfg = _clean({
        "sampler": sample.get("sampler") or ("ddpm" if model["arch"] in ("sdxl", "sd15", "sd1") else "flowmatch"),
        "sample_every": steps,          # only at the end (+ skip_first_sample below)
        "sample_start_step": 0,
        "width": int(sample.get("width") or 1024),
        "height": int(sample.get("height") or 1024),
        "prompts": list(sample.get("prompts") or []),
        "neg": "",
        "seed": 42,
        "walk_seed": True,
        "guidance_scale": float(sample.get("guidance_scale", 4)),
        "sample_steps": int(sample.get("sample_steps") or 25),
        "num_frames": int(sample["num_frames"]) if sample.get("num_frames") else None,
        "fps": int(sample["fps"]) if sample.get("fps") else None,
    })

    train_cfg = _clean({
        "batch_size": 1,
        "steps": steps,
        "gradient_accumulation": 1,
        "train_unet": True,
        "train_text_encoder": False,
        "gradient_checkpointing": True,
        "noise_scheduler": train.get("noise_scheduler") or ("ddpm" if model["arch"] in ("sdxl", "sd15", "sd1") else "flowmatch"),
        "optimizer": train.get("optimizer") or "adamw8bit",
        "lr": train["lr"],
        "timestep_type": train.get("timestep_type"),
        "cache_text_embeddings": train.get("cache_text_embeddings"),
        "unload_text_encoder": train.get("unload_text_encoder"),
        "dtype": "bf16",
        "skip_first_sample": True,
        "disable_sampling": not sampling_enabled,
        "ema_config": {"use_ema": True, "ema_decay": 0.99} if train.get("ema") else None,
        "optimizer_params": train.get("optimizer_params"),
        "switch_boundary_every": train.get("switch_boundary_every"),
    })

    network_cfg = _clean({
        "type": "lora",
        "linear": train["rank"],
        "linear_alpha": int(train.get("alpha") or train["rank"]),
        "network_kwargs": {"ignore_if_contains": list(network["ignore_if_contains"])} if network.get("ignore_if_contains") else None,
    })

    model_cfg = _clean({
        "name_or_path": model["name_or_path"],
        "arch": model["arch"],
        "quantize": model.get("quantize"),
        "qtype": model.get("qtype"),
        "quantize_te": model.get("quantize_te"),
        "qtype_te": model.get("qtype_te"),
        "low_vram": model.get("low_vram"),
        "assistant_lora_path": model.get("assistant_lora_path"),
        "extras_name_or_path": model.get("extras_name_or_path"),
        "model_kwargs": model.get("model_kwargs"),
    })

    process = _clean({
        "type": "sd_trainer",
        "training_folder": os.path.join(workdir, "output"),
        "device": "cuda:0",
        "trigger_word": job.get("trigger_word"),
        "performance_log_every": 0,
        "network": network_cfg,
        "save": {"dtype": "float16", "save_every": steps, "max_step_saves_to_keep": 1, "push_to_hub": False},
        "datasets": [dataset_cfg],
        "train": train_cfg,
        "model": model_cfg,
        "sample": sample_cfg,
    })

    return {
        "job": "extension",
        "config": {"name": name, "process": [process]},
        "meta": {"name": "[name]", "version": "1.0"},
    }


# Caption files -----------------------------------------------------------------------------------

def caption_for(item: dict, trigger_word: str | None, default_caption: str | None) -> str:
    """The caption written next to a dataset file. The trigger word is always present
    (ai-toolkit only substitutes `[trigger]`; it does not prepend)."""
    cap = (item.get("caption") or "").strip() or (default_caption or "").strip()
    trig = (trigger_word or "").strip()
    if trig:
        if "[trigger]" in cap:
            cap = cap.replace("[trigger]", trig)
        elif trig.lower() not in cap.lower():
            cap = f"{trig}, {cap}" if cap else trig
    return cap or (trig or "")


# Progress parsing ------------------------------------------------------------------------------

STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
TQDM_TIME_RE = re.compile(r"\[(\d+(?::\d+){1,2})<(\d+(?::\d+){1,2}|\?)")
LOSS_RE = re.compile(r"loss:\s*([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)", re.IGNORECASE)
PHASE_HINTS = (
    ("Generating Images", "sampling"),
    ("Generating baseline", "sampling"),
    ("Caching latents", "caching"),
    ("Caching text", "caching"),
    ("Loading", "loading"),
    ("Saving", "saving"),
)


def _hms_to_seconds(s: str) -> int | None:
    if not s or s == "?":
        return None
    parts = [int(p) for p in s.split(":")]
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


def parse_progress_line(line: str, total_steps: int) -> dict[str, Any] | None:
    """Extract {step, total, remaining_s, loss} from an ai-toolkit/tqdm line, or a {phase} hint."""
    m = STEP_RE.search(line)
    if m:
        step, total = int(m.group(1)), int(m.group(2))
        if total == total_steps and 0 <= step <= total:
            out: dict[str, Any] = {"step": step, "total": total}
            t = TQDM_TIME_RE.search(line)
            if t:
                out["elapsed_s"] = _hms_to_seconds(t.group(1))
                out["remaining_s"] = _hms_to_seconds(t.group(2))
            lm = LOSS_RE.search(line)
            if lm:
                try:
                    out["loss"] = float(lm.group(1))
                except ValueError:
                    pass
            return out
    for hint, phase in PHASE_HINTS:
        if hint.lower() in line.lower():
            return {"phase": phase}
    return None
