"""R2 / S3-compatible storage for the LoRA trainer.

Same env contract as the ComfyUI workers (base/lib/r2.py):
  BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY, R2_BUCKET_NAME,
  optional R2_INPUT_BUCKET_NAME (defaults to R2_BUCKET_NAME) and BUCKET_PROFILES (JSON
  allowlist of named targets; the job picks one with input.bucket_profile).

Unlike the ComfyUI workers there is no base64 fallback: a LoRA is tens to hundreds of
MB, so R2 is required and the handler fails the job early when it is not configured.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import boto3
from botocore.config import Config


@dataclass(frozen=True)
class BucketTarget:
    bucket: str
    input_bucket: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    profile: str | None = None


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def load_profiles() -> dict[str, dict]:
    raw = _env("BUCKET_PROFILES")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"BUCKET_PROFILES is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("BUCKET_PROFILES must be a JSON object of {name: {bucket: ...}}")
    for name, cfg in data.items():
        if not isinstance(cfg, dict) or not cfg.get("bucket"):
            raise RuntimeError(f"BUCKET_PROFILES['{name}'] needs a 'bucket'")
        has_key = bool(cfg.get("access_key_id"))
        has_secret = bool(cfg.get("secret_access_key"))
        if has_key != has_secret:
            raise RuntimeError(f"BUCKET_PROFILES['{name}'] must set both access_key_id and secret_access_key or neither")
    return data


def resolve_bucket_target(profile_name: str | None) -> BucketTarget:
    """Pick the bucket a job reads inputs from and writes outputs to."""
    endpoint_url = _env("BUCKET_ENDPOINT_URL")
    access_key_id = _env("BUCKET_ACCESS_KEY_ID")
    secret = _env("BUCKET_SECRET_ACCESS_KEY")
    default_bucket = _env("R2_BUCKET_NAME")
    default_input_bucket = _env("R2_INPUT_BUCKET_NAME") or default_bucket

    if profile_name:
        profiles = load_profiles()
        cfg = profiles.get(profile_name)
        if cfg is None:
            raise ValueError(f"Unknown bucket_profile '{profile_name}'")
        return BucketTarget(
            bucket=cfg["bucket"],
            input_bucket=cfg.get("input_bucket") or cfg["bucket"],
            endpoint_url=cfg.get("endpoint_url") or endpoint_url,
            access_key_id=cfg.get("access_key_id") or access_key_id,
            secret_access_key=cfg.get("secret_access_key") or secret,
            profile=profile_name,
        )

    if not (endpoint_url and access_key_id and secret and default_bucket):
        raise ValueError(
            "R2 is not configured on this endpoint: set BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, "
            "BUCKET_SECRET_ACCESS_KEY and R2_BUCKET_NAME"
        )
    return BucketTarget(
        bucket=default_bucket,
        input_bucket=default_input_bucket,
        endpoint_url=endpoint_url,
        access_key_id=access_key_id,
        secret_access_key=secret,
    )


def make_client(target: BucketTarget):
    return boto3.client(
        "s3",
        endpoint_url=target.endpoint_url,
        aws_access_key_id=target.access_key_id,
        aws_secret_access_key=target.secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def download_to_file(client, bucket: str, key: str, dest_path: str) -> int:
    """Download one object to dest_path (atomic rename). Returns the byte size."""
    tmp = dest_path + ".part"
    client.download_file(bucket, key, tmp)
    size = os.path.getsize(tmp)
    os.replace(tmp, dest_path)
    return size


_CONTENT_TYPES = {
    ".safetensors": "application/octet-stream",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".txt": "text/plain",
    ".yaml": "text/yaml",
    ".log": "text/plain",
}


def upload_file(client, bucket: str, key: str, path: str, content_type: str | None = None) -> int:
    ext = os.path.splitext(key)[1].lower()
    ctype = content_type or _CONTENT_TYPES.get(ext, "application/octet-stream")
    client.upload_file(path, bucket, key, ExtraArgs={"ContentType": ctype})
    return os.path.getsize(path)
