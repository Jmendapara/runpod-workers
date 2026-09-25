"""R2 / S3-compatible upload + presigned URLs, with per-job bucket selection.

Where a job's media lives is a *bucket target*: endpoint URL, credentials, the
output bucket and the input bucket. A job gets its target from one of two places:

  * the endpoint's own env (BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID,
    BUCKET_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_INPUT_BUCKET_NAME) — the
    default for every job that does not name a profile (today's behaviour); or
  * a named entry of the BUCKET_PROFILES env (a JSON object), selected by the
    job's `bucket_profile` input. One endpoint can then serve several apps and
    environments: each profile is an allowlisted bucket, with its own
    credentials only when the endpoint's default key cannot reach it.

    BUCKET_PROFILES='{"pd-qa": {"bucket": "pd-qa-media"},
                      "osg-dev": {"bucket": "osg-media-dev", "input_bucket": "osg-uploads-dev",
                                  "access_key_id": "...", "secret_access_key": "..."}}'

An unknown profile fails the job before ComfyUI runs; a malformed
BUCKET_PROFILES kills the worker at boot (handler.py). `resolve_bucket_target(None)`
returns None when BUCKET_ENDPOINT_URL is unset → base64 responses. A named
profile always implies R2 upload.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass

PROFILES_ENV = "BUCKET_PROFILES"

PRESIGN_TTL_SECONDS = 7 * 24 * 3600  # 7 days

# Generated media is immutable per key, so let browsers cache it for a year.
# Without this, players re-download the object on every remount (e.g. navigating
# back to the Library). Set at upload so every model gets it for free.
CACHE_CONTROL = "public, max-age=31536000"

_PROFILE_KEYS = frozenset({"bucket", "input_bucket", "endpoint_url", "access_key_id", "secret_access_key"})
_CRED_KEYS = ("access_key_id", "secret_access_key")


@dataclass(frozen=True)
class BucketTarget:
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    input_bucket: str
    profile: str | None = None  # None → the endpoint's default env target

    def describe(self) -> str:
        """Log-safe summary — never includes credentials."""
        source = f"profile '{self.profile}'" if self.profile else "endpoint default"
        return f"{source}: bucket={self.bucket} input_bucket={self.input_bucket}"


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


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    value = value.strip() if isinstance(value, str) else None
    return value or None


def load_bucket_profiles(raw: str | None = None) -> dict[str, dict]:
    """Parse BUCKET_PROFILES (or `raw`). Returns {} when unset.

    Raises ValueError on anything malformed so the handler can refuse to boot —
    a typo here must never silently route media to the wrong bucket.
    """
    if raw is None:
        raw = os.environ.get(PROFILES_ENV)
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{PROFILES_ENV} is not valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"{PROFILES_ENV} must be a JSON object mapping profile name → settings")

    for name, profile in parsed.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{PROFILES_ENV} has an empty profile name")
        if not isinstance(profile, dict):
            raise ValueError(f"{PROFILES_ENV}['{name}'] must be an object")
        unknown = sorted(set(profile) - _PROFILE_KEYS)
        if unknown:
            raise ValueError(f"{PROFILES_ENV}['{name}'] has unknown keys: {unknown}")
        bucket = profile.get("bucket")
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError(f"{PROFILES_ENV}['{name}'] needs a non-empty 'bucket'")
        for key in _PROFILE_KEYS - {"bucket"}:
            value = profile.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{PROFILES_ENV}['{name}'].{key} must be a non-empty string when set")
        # Credentials are all-or-none per profile: mixing one key from the profile
        # with the other from the endpoint env can only produce auth failures.
        if bool(profile.get(_CRED_KEYS[0])) != bool(profile.get(_CRED_KEYS[1])):
            raise ValueError(
                f"{PROFILES_ENV}['{name}'] must set both access_key_id and secret_access_key, or neither"
            )
    return parsed


def resolve_bucket_target(profile_name: str | None) -> BucketTarget | None:
    """Pick the bucket target for a job.

    profile_name None → the endpoint's env target (None when no R2 is configured,
    which means base64 responses). Otherwise the named BUCKET_PROFILES entry,
    inheriting endpoint URL / credentials from the env when the profile omits them.
    Raises ValueError with a job-safe message on any misconfiguration.
    """
    endpoint = _env("BUCKET_ENDPOINT_URL")
    access = _env("BUCKET_ACCESS_KEY_ID")
    secret = _env("BUCKET_SECRET_ACCESS_KEY")

    if profile_name is None:
        if not endpoint:
            return None
        bucket = _env("R2_BUCKET_NAME")
        if not bucket:
            raise ValueError(
                "BUCKET_ENDPOINT_URL is set but R2_BUCKET_NAME is missing. "
                "Either unset BUCKET_ENDPOINT_URL (base64 mode) or set R2_BUCKET_NAME."
            )
        if not access or not secret:
            raise ValueError(
                "R2 credentials not configured. "
                "BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, and "
                "BUCKET_SECRET_ACCESS_KEY must all be set."
            )
        return BucketTarget(endpoint, access, secret, bucket, _env("R2_INPUT_BUCKET_NAME") or bucket)

    profiles = load_bucket_profiles()
    profile = profiles.get(profile_name)
    if profile is None:
        allowed = ", ".join(sorted(profiles)) or "none"
        raise ValueError(f"Unknown bucket_profile '{profile_name}'. This endpoint allows: {allowed}")

    endpoint = (profile.get("endpoint_url") or "").strip() or endpoint
    if profile.get("access_key_id"):
        access = profile["access_key_id"].strip()
        secret = profile["secret_access_key"].strip()
    if not endpoint or not access or not secret:
        raise ValueError(
            f"bucket_profile '{profile_name}' has no complete R2 credentials. Set "
            "endpoint_url / access_key_id / secret_access_key in the profile or the BUCKET_* env."
        )
    bucket = profile["bucket"].strip()
    input_bucket = (profile.get("input_bucket") or "").strip() or bucket
    return BucketTarget(endpoint, access, secret, bucket, input_bucket, profile=profile_name)


def make_s3_client(target: BucketTarget):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=target.endpoint_url,
        aws_access_key_id=target.access_key_id,
        aws_secret_access_key=target.secret_access_key,
    )


class Uploader:
    def __init__(self, target: BucketTarget):
        self.target = target
        self.bucket = target.bucket
        self._client = make_s3_client(target)

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


def make_uploader(target: BucketTarget | None) -> Uploader | None:
    """Return an Uploader for the job's bucket target, or None for base64 mode."""
    if target is None:
        return None
    return Uploader(target)
