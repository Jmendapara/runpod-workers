#!/usr/bin/env python3
"""POST a smoke-test input to a RunPod serverless endpoint and assert the response shape.

Usage:
  python tests/run_smoke.py <model-name> <endpoint-id> [--api-key KEY]

Inputs come from tests/smoke/<model-name>.json. Asserts:
  - HTTP 200, no top-level 'error'
  - Output arrays for the model's declared output.type are non-empty
  - Magic-byte check on decoded base64 payloads (or HEAD on s3_urls)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

MAGIC_BYTES = {
    "image": [b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"RIFF"],
    "video": [b"ftyp", b"\x1aE\xdf\xa3"],  # ftyp at offset 4-ish, EBML for webm
    "audio": [b"RIFF", b"fLaC", b"OggS", b"ID3"],
}


def magic_match(file_bytes: bytes, kinds: list[str]) -> bool:
    for kind in kinds:
        for magic in MAGIC_BYTES.get(kind, []):
            if magic == b"ftyp" and len(file_bytes) >= 12 and b"ftyp" in file_bytes[:12]:
                return True
            if file_bytes.startswith(magic):
                return True
    return False


def expected_keys(output_type: str) -> list[tuple[str, list[str]]]:
    """Return list of (response_key, allowed_magic_kinds) we expect to see populated."""
    m = {
        "image":              [("images", ["image"])],
        "video":              [("videos", ["video"])],
        "audio":              [("audio",  ["audio"])],
        "image+audio":        [("images", ["image"]), ("audio", ["audio"])],
        "image+video":        [("videos", ["video"])],  # images often empty for video workflows
        "image+video+gifs":   [("videos", ["video"])],
    }
    return m.get(output_type, [])


def assert_item(item: dict, magic_kinds: list[str]) -> None:
    kind = item.get("type")
    if kind == "base64":
        b = base64.b64decode(item["data"])
        assert magic_match(b, magic_kinds), f"magic-byte mismatch for {item.get('filename')}"
    elif kind == "s3_url":
        head = requests.head(item["data"], timeout=15)
        assert head.status_code in (200, 206), f"presigned URL HEAD returned {head.status_code}"
    else:
        raise AssertionError(f"unknown item type: {kind}")


def run(model: str, endpoint_id: str, api_key: str, timeout_s: int = 600) -> int:
    smoke_path = REPO_ROOT / "tests" / "smoke" / f"{model}.json"
    if not smoke_path.exists():
        print(f"ERROR: no smoke input at {smoke_path}", file=sys.stderr)
        return 2

    with open(smoke_path) as f:
        payload = json.load(f)

    model_yaml_path = REPO_ROOT / "models" / model / "model.yaml"
    with open(model_yaml_path) as f:
        cfg = yaml.safe_load(f)
    out_type = cfg["output"]["type"]
    print(f"Model: {model}  output.type: {out_type}")

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    print(f"POST {url}")
    resp = requests.post(url, headers=headers, data=json.dumps({"input": payload}), timeout=timeout_s)
    print(f"  status={resp.status_code}")
    resp.raise_for_status()
    body = resp.json()

    if body.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        job_id = body["id"]
        poll_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(2)
            poll = requests.get(poll_url, headers=headers, timeout=30).json()
            if poll.get("status") in ("COMPLETED", "FAILED"):
                body = poll
                break
        else:
            print("ERROR: timed out waiting for completion", file=sys.stderr)
            return 3

    if body.get("status") == "FAILED":
        print(f"ERROR: endpoint returned FAILED: {json.dumps(body, indent=2)}", file=sys.stderr)
        return 4

    output = body.get("output") or {}
    if "error" in output:
        print(f"ERROR: handler returned: {output['error']}", file=sys.stderr)
        if output.get("details"):
            print(f"  details: {output['details']}", file=sys.stderr)
        return 5

    failures: list[str] = []
    for key, magic_kinds in expected_keys(out_type):
        items = output.get(key) or []
        if not items:
            failures.append(f"output.{key} is empty (expected non-empty for {out_type})")
            continue
        for item in items:
            try:
                assert_item(item, magic_kinds)
            except AssertionError as exc:
                failures.append(f"output.{key}: {exc}")

    if failures:
        print("ASSERTION FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 6

    print(f"✓ smoke test passed for {model}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name (matches models/<name>/)")
    parser.add_argument("endpoint_id", help="RunPod serverless endpoint ID")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not args.api_key:
        print("ERROR: set RUNPOD_API_KEY or pass --api-key", file=sys.stderr)
        return 64
    return run(args.model, args.endpoint_id, args.api_key, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
