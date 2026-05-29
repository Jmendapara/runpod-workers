"""R2 / S3-compatible upload + presigned URLs. Env-var toggled.

`make_uploader()` returns None when BUCKET_ENDPOINT_URL is unset → handler
returns base64. When set, returns an Uploader that uploads and presigns.
"""
from __future__ import annotations

import os
import uuid


PRESIGN_TTL_SECONDS = 7 * 24 * 3600  # 7 days

# Generated media is immutable per key, so let browsers cache it for a year.
# Without this, players re-download the object on every remount (e.g. navigating
# back to the Library). Set at upload so every model gets it for free.
CACHE_CONTROL = "public, max-age=31536000"


def _guess_content_type(ext: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".gif": "image/gif",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
    }.get(ext.lower(), "application/octet-stream")


def make_s3_client():
    import boto3

    endpoint = os.environ.get("BUCKET_ENDPOINT_URL")
    access = os.environ.get("BUCKET_ACCESS_KEY_ID")
    secret = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
    if not endpoint or not access or not secret:
        raise ValueError(
            "R2 credentials not configured. "
            "BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, and "
            "BUCKET_SECRET_ACCESS_KEY must all be set."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
    )


class Uploader:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self._client = make_s3_client()

    def upload_returning_key(self, file_bytes: bytes, filename: str, job_id: str, uid: str | None = None) -> tuple[str, str]:
        """Upload and return (object_key, presigned_get_url).

        Use when the caller needs the raw R2 key (e.g. derived poster/preview
        assets that the app stores as keys and re-signs on read).
        """
        ext = os.path.splitext(filename)[1] or ".bin"
        leaf = f"{str(uuid.uuid4())[:8]}{ext}"
        key = f"users/{uid}/generations/{leaf}" if uid else f"{job_id}/{leaf}"

        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=file_bytes,
            ContentType=_guess_content_type(ext),
            CacheControl=CACHE_CONTROL,
        )
        url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=PRESIGN_TTL_SECONDS,
        )
        return key, url

    def upload(self, file_bytes: bytes, filename: str, job_id: str, uid: str | None = None) -> str:
        _key, url = self.upload_returning_key(file_bytes, filename, job_id, uid=uid)
        return url


def make_uploader() -> Uploader | None:
    """Return an Uploader if R2 env is configured, else None (base64 mode)."""
    if not os.environ.get("BUCKET_ENDPOINT_URL"):
        return None
    bucket = os.environ.get("R2_BUCKET_NAME")
    if not bucket:
        raise ValueError(
            "BUCKET_ENDPOINT_URL is set but R2_BUCKET_NAME is missing. "
            "Either unset BUCKET_ENDPOINT_URL (base64 mode) or set R2_BUCKET_NAME."
        )
    return Uploader(bucket)
