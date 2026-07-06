"""Universal RunPod ComfyUI worker handler.

Reads /etc/worker/model.yaml at import time → exits non-zero with a clear
message if validation fails. Once loaded, handles workflow submission, output
collection, and R2/base64 response shaping based on the model's config.
"""
from __future__ import annotations

import os
import sys
import traceback
import uuid

# /opt/worker is on PYTHONPATH (set in the base Dockerfile)
from lib.config import load_model_config
from lib.comfy_client import ComfyClient
from lib.collectors import build_collectors
from lib.inputs import process_inline_images, process_r2_inputs, process_r2_loras, validate_input
from lib.r2 import make_uploader

import requests
import runpod
import websocket

from network_volume import is_network_volume_debug_enabled, run_network_volume_diagnostics


# Module-load-time config + collectors (fail-fast on bad model.yaml)
MODEL_CFG = load_model_config()
COLLECTORS = build_collectors(MODEL_CFG.output)

# WebSocket trace toggle
if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    websocket.enableTrace(True)

REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"


def handler(job):
    if is_network_volume_debug_enabled():
        run_network_volume_diagnostics()

    job_input = job["input"]
    job_id = job["id"]

    validated, error_msg = validate_input(job_input)
    if error_msg:
        return {"error": error_msg}

    workflow = validated["workflow"]
    r2_inputs = validated["r2_inputs"]
    images = validated["images"]
    r2_loras = validated["r2_loras"]
    uid = validated.get("uid")
    comfy_org_api_key = validated.get("comfy_org_api_key")

    client = ComfyClient()
    if not client.check_server():
        return {"error": f"ComfyUI server ({client.host}) not reachable after retries."}

    try:
        process_r2_inputs(workflow, r2_inputs)
    except Exception as exc:
        print(f"worker-comfyui - R2 input download failed: {exc}", flush=True)
        traceback.print_exc()
        return {"error": f"Failed to download R2 inputs: {exc}"}

    try:
        process_inline_images(workflow, images)
    except Exception as exc:
        print(f"worker-comfyui - Inline image decode failed: {exc}", flush=True)
        traceback.print_exc()
        return {"error": f"Failed to process inline images: {exc}"}

    # Custom LoRAs must be on disk (cache hit or fresh download) BEFORE the
    # workflow is queued — LoraLoader nodes resolve lora_name at queue time.
    try:
        process_r2_loras(r2_loras)
    except Exception as exc:
        print(f"worker-comfyui - LoRA download failed: {exc}", flush=True)
        traceback.print_exc()
        return {"error": f"Failed to download LoRA: {exc}"}

    try:
        uploader = make_uploader()
    except ValueError as exc:
        return {"error": str(exc)}

    client_id = str(uuid.uuid4())

    try:
        prompt_id = client.queue_prompt(workflow, client_id, comfy_org_api_key)
        print(f"worker-comfyui - Queued workflow with ID: {prompt_id}", flush=True)
    except ValueError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"Error queuing workflow: {exc}"}

    try:
        exec_errors = client.wait_for_completion(client.ws_url(client_id), prompt_id)
    except websocket.WebSocketException as exc:
        traceback.print_exc()
        return {"error": f"ComfyUI communication lost: {exc}"}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Unexpected error waiting for completion: {exc}"}

    try:
        history = client.get_history(prompt_id)
    except requests.RequestException as exc:
        return {"error": f"Failed to fetch history: {exc}"}

    if prompt_id not in history:
        details = list(exec_errors) + [f"Prompt ID {prompt_id} not found in history."]
        return {"error": "Job processing failed", "details": details}

    final = COLLECTORS.harvest(
        history.get(prompt_id, {}),
        comfy_client=client,
        job_id=job_id,
        uploader=uploader,
        uid=uid,
    )

    if exec_errors:
        final.setdefault("errors", []).extend(exec_errors)

    has_output = any(
        isinstance(v, list) and v
        for k, v in final.items()
        if k != "errors"
    )
    if not has_output:
        if final.get("errors"):
            return {"error": "Job processing failed", "details": final["errors"]}
        final["status"] = "success_no_output"

    print(
        f"worker-comfyui - Job done. Returning keys: "
        f"{[k for k in final.keys() if k != 'errors']}",
        flush=True,
    )
    return final


if __name__ == "__main__":
    print(
        f"worker-comfyui - Starting handler for model={MODEL_CFG.name}, "
        f"output.type={MODEL_CFG.output.type}",
        flush=True,
    )
    runpod.serverless.start({"handler": handler, "refresh_worker": REFRESH_WORKER})
