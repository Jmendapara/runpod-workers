"""Materialize r2_inputs + inline images into /comfyui/input/ and rewrite the
workflow to reference local filenames.

Same pattern every existing handler used, consolidated here.
"""
from __future__ import annotations

import base64
import json
import os

COMFY_INPUT_DIR = "/comfyui/input"


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

    if not isinstance(r2_inputs, list):
        return None, "'r2_inputs' must be a list"
    if not isinstance(images, list):
        return None, "'images' must be a list"

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
        "uid": uid,
        "comfy_org_api_key": job_input.get("comfy_org_api_key"),
    }, None


def process_r2_inputs(workflow: dict, r2_inputs: list[dict]) -> bool:
    """Download each R2 input into /comfyui/input/ and rewrite the workflow.

    Returns True iff any downloaded video lacked an audio stream and had silent
    audio muxed in. Callers use this to adjust output handling (e.g. dropping
    VHS's -audio.mp4 sidecar in favor of the silent .mp4 variant).
    """
    if not r2_inputs:
        return False

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
    any_silenced = False
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

        if ensure_audio_track(local_path):
            any_silenced = True

    return any_silenced


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
