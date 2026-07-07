"""Materialize r2_inputs + inline images into /comfyui/input/ and rewrite the
workflow to reference local filenames.

Same pattern every existing handler used, consolidated here.

Also handles r2_loras: custom (user-uploaded) LoRA files downloaded from R2
into ComfyUI's loras dir before the workflow is queued, with a container-local
cache keyed by filename (filenames are globally unique and immutable).
"""
from __future__ import annotations

import base64
import json
import os
import uuid

COMFY_INPUT_DIR = "/comfyui/input"
# The baked loras dir is already a registered ComfyUI `loras` search path, and
# ComfyUI's folder-list cache invalidates on the dir's mtime — files written
# here become resolvable by LoraLoader nodes without a restart.
COMFY_LORA_DIR = "/comfyui/models/loras"

LORA_SUFFIX = ".safetensors"


def validate_input(job_input) -> tuple[dict | None, str | None]:
    """Validate and normalize the job input. Returns (data, error_message)."""
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    r2_inputs = job_input.get("r2_inputs") or []
    images = job_input.get("images") or []
    r2_loras = job_input.get("r2_loras") or []

    if not isinstance(r2_inputs, list):
        return None, "'r2_inputs' must be a list"
    if not isinstance(images, list):
        return None, "'images' must be a list"
    if not isinstance(r2_loras, list):
        return None, "'r2_loras' must be a list"

    for i, entry in enumerate(r2_inputs):
        if not isinstance(entry, dict):
            return None, f"r2_inputs[{i}] must be an object"
        for req in ("node_id", "input_field", "r2_key"):
            if req not in entry:
                return None, f"r2_inputs[{i}] missing required field '{req}'"

    for i, entry in enumerate(images):
        if not isinstance(entry, dict):
            return None, f"images[{i}] must be an object"
        for req in ("node_id", "input_field", "name", "image"):
            if req not in entry:
                return None, f"images[{i}] missing required field '{req}'"

    lora_key_by_filename: dict[str, str] = {}
    for i, entry in enumerate(r2_loras):
        if not isinstance(entry, dict):
            return None, f"r2_loras[{i}] must be an object"
        for req in ("r2_key", "filename"):
            if not isinstance(entry.get(req), str) or not entry[req]:
                return None, f"r2_loras[{i}] missing required string field '{req}'"
        filename = entry["filename"]
        # The filename is written into the loras dir verbatim — it must be a
        # bare .safetensors basename (path-traversal guard).
        if (
            os.path.basename(filename) != filename
            or not filename.endswith(LORA_SUFFIX)
            or len(filename) <= len(LORA_SUFFIX)
        ):
            return None, f"r2_loras[{i}] filename must be a bare .safetensors basename"
        size_bytes = entry.get("size_bytes")
        if size_bytes is not None and (
            not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0
        ):
            return None, f"r2_loras[{i}] size_bytes must be a positive integer"
        prior_key = lora_key_by_filename.get(filename)
        if prior_key is not None and prior_key != entry["r2_key"]:
            return None, f"r2_loras[{i}] filename '{filename}' maps to conflicting r2_keys"
        lora_key_by_filename[filename] = entry["r2_key"]

    seen: set[tuple[str, str]] = set()
    for arr_name, arr in (("r2_inputs", r2_inputs), ("images", images)):
        for i, entry in enumerate(arr):
            key = (str(entry["node_id"]), entry["input_field"])
            if key in seen:
                return None, (
                    f"{arr_name}[{i}] targets node_id={key[0]}, input_field={key[1]} "
                    f"which is already targeted by an earlier entry"
                )
            seen.add(key)

    uid = job_input.get("uid")
    if uid is not None:
        if not isinstance(uid, str) or not uid.strip():
            return None, "'uid' must be a non-empty string"
        if "/" in uid:
            return None, "'uid' must not contain '/'"

    return {
        "workflow": workflow,
        "r2_inputs": r2_inputs,
        "images": images,
        "r2_loras": r2_loras,
        "uid": uid,
        "comfy_org_api_key": job_input.get("comfy_org_api_key"),
    }, None


def process_r2_inputs(workflow: dict, r2_inputs: list[dict]) -> None:
    """Download each R2 input into /comfyui/input/ and rewrite the workflow.

    Video inputs whose audio is missing or shorter than the video get repaired
    in place (see ensure_audio_track): a silent stereo track is muxed in, or a
    short track is padded with trailing silence to the video length. This keeps
    downstream nodes that assume full-length audio (VHS NormalizeAudioLoudness,
    LTX audio VAE, TrimAudioDuration) from failing on audio-less or audio-short
    source clips. The injected silence is inaudible.
    """
    if not r2_inputs:
        return

    from .r2 import make_s3_client
    from .ffmpeg_helpers import ensure_audio_track

    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)

    input_bucket = os.environ.get("R2_INPUT_BUCKET_NAME") or os.environ.get("R2_BUCKET_NAME")
    if not input_bucket:
        raise ValueError(
            "No input bucket configured. Set R2_INPUT_BUCKET_NAME or R2_BUCKET_NAME."
        )

    s3 = make_s3_client()
    print(
        f"worker-comfyui - Downloading {len(r2_inputs)} R2 input(s) from '{input_bucket}'...",
        flush=True,
    )
    for entry in r2_inputs:
        node_id = str(entry["node_id"])
        field = entry["input_field"]
        key = entry["r2_key"]

        if node_id not in workflow:
            raise ValueError(f"r2_inputs references node_id '{node_id}' not in the workflow")
        if "inputs" not in workflow[node_id]:
            raise ValueError(f"Workflow node '{node_id}' has no 'inputs' dict")

        filename = os.path.basename(key)
        local_path = os.path.join(COMFY_INPUT_DIR, filename)
        print(f"worker-comfyui - R2: {key} -> {local_path} (node {node_id}.{field})", flush=True)
        s3.download_file(input_bucket, key, local_path)
        workflow[node_id]["inputs"][field] = filename

        ensure_audio_track(local_path)


def process_r2_loras(r2_loras: list[dict]) -> None:
    """Download custom (user-uploaded) LoRAs from R2 into ComfyUI's loras dir.

    Unlike r2_inputs there is no workflow rewrite: the server already set each
    LoraLoader node's lora_name to the entry's filename; this just guarantees
    the file exists on disk before the workflow is queued.

    Cache: filenames are `{loraId}.safetensors` — globally unique and immutable
    once uploaded (renames never touch the R2 object; re-upload = new loraId).
    A local file with the expected size is therefore always current, letting
    warm workers skip the (up to 500MB) download entirely. Downloads land in a
    temp path and are moved into place atomically so a crashed download can
    never leave a truncated file that would pass the size check.
    """
    if not r2_loras:
        return

    from .r2 import make_s3_client

    os.makedirs(COMFY_LORA_DIR, exist_ok=True)

    input_bucket = os.environ.get("R2_INPUT_BUCKET_NAME") or os.environ.get("R2_BUCKET_NAME")
    if not input_bucket:
        raise ValueError(
            "No input bucket configured. Set R2_INPUT_BUCKET_NAME or R2_BUCKET_NAME."
        )

    s3 = None
    for entry in r2_loras:
        key = entry["r2_key"]
        filename = entry["filename"]
        size_bytes = entry.get("size_bytes")
        local_path = os.path.join(COMFY_LORA_DIR, filename)

        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            if not size_bytes or local_size == size_bytes:
                print(
                    f"worker-comfyui - LoRA cache hit: {filename} ({local_size} bytes)",
                    flush=True,
                )
                continue
            print(
                f"worker-comfyui - LoRA size mismatch for {filename} "
                f"(local {local_size} != expected {size_bytes}) - re-downloading",
                flush=True,
            )

        if s3 is None:
            s3 = make_s3_client()
        tmp_path = f"{local_path}.part-{uuid.uuid4().hex}"
        print(f"worker-comfyui - LoRA: {key} -> {local_path}", flush=True)
        try:
            s3.download_file(input_bucket, key, tmp_path)
            os.replace(tmp_path, local_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def process_inline_images(workflow: dict, images: list[dict]) -> None:
    """Decode each base64 image, write to /comfyui/input/, rewrite the workflow."""
    if not images:
        return

    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    print(f"worker-comfyui - Decoding {len(images)} inline image(s)...", flush=True)
    for entry in images:
        node_id = str(entry["node_id"])
        field = entry["input_field"]
        name = entry["name"]
        image_data = entry["image"]

        if node_id not in workflow:
            raise ValueError(f"images references node_id '{node_id}' not in the workflow")
        if "inputs" not in workflow[node_id]:
            raise ValueError(f"Workflow node '{node_id}' has no 'inputs' dict")

        if "," in image_data and image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]

        try:
            file_bytes = base64.b64decode(image_data)
        except Exception as exc:
            raise ValueError(f"images entry node {node_id}.{field}: invalid base64: {exc}")

        filename = os.path.basename(name)
        local_path = os.path.join(COMFY_INPUT_DIR, filename)
        with open(local_path, "wb") as f:
            f.write(file_bytes)
        print(
            f"worker-comfyui - inline: {filename} ({len(file_bytes)} bytes) -> "
            f"{local_path} (node {node_id}.{field})",
            flush=True,
        )
        workflow[node_id]["inputs"][field] = filename
