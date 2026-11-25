"""Input contract of the TRELLIS.2-4B worker. Pure Python (no torch), so it is unit-tested on a laptop.

job.input:
  image_r2_key       str            the picture, read from R2_INPUT_BUCKET_NAME (default R2_BUCKET_NAME)
  image_r2_keys      [str], 1..4    several pictures of the SAME object from different sides (front, back,
                                    a side); the first is the main view. One of the two is required.
  together_steps     0..50: with several pictures, how many of the last sampler steps of each stage use
                     all of them at once; the steps before take them in turn (multi_image.py). Default:
                     0 for two pictures, 4 for three or four. Ignored with one picture.
  sampler_steps      4..50 (default: pipeline.json's 12): with several pictures, the steps of every stage's
                     sampler. Ignored with one picture.
  quality            'standard' (1024³ cascade, the upstream default) | 'high' (1536³ cascade)
  seed               int 0..2^31-1 (default 42, the pipeline's own default)
  texture_size       1024 | 2048 | 4096 (default 2048, the demo's)
  decimation_target  faces kept in the GLB, 50k..1M (default 500k, the demo's)
  turntable          bool (default true): also render a 360° MP4 of the model
  uid                optional: outputs go under users/{uid}/generations/ (else {job_id}/)
"""
from __future__ import annotations

from dataclasses import dataclass

PIPELINE_TYPES = {"standard": "1024_cascade", "high": "1536_cascade"}
TEXTURE_SIZES = (1024, 2048, 4096)
MIN_FACES, MAX_FACES = 50_000, 1_000_000
SEED_MAX = 2_147_483_647
MAX_KEY_LENGTH = 512
MAX_IMAGES = 4
MAX_TOGETHER_STEPS = 50
MIN_SAMPLER_STEPS, MAX_SAMPLER_STEPS = 4, 50


@dataclass(frozen=True)
class JobSpec:
    image_r2_keys: tuple[str, ...]
    together_steps: int | None
    sampler_steps: int | None
    pipeline_type: str
    quality: str
    seed: int
    texture_size: int
    decimation_target: int
    turntable: bool
    uid: str | None

    @property
    def image_r2_key(self) -> str:
        """The main picture (the only one for a single-picture job)."""
        return self.image_r2_keys[0]


class InputError(ValueError):
    """The request is malformed (shown to the caller as the job error)."""


def _int(raw, name: str) -> int:
    if isinstance(raw, bool):
        raise InputError(f"{name} must be a number")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number") from None
    if float(raw) != value:
        raise InputError(f"{name} must be a whole number")
    return value


def _key(key) -> str:
    if not isinstance(key, str) or not key.strip() or len(key) > MAX_KEY_LENGTH or ".." in key or key.startswith("/"):
        raise InputError("image_r2_key must be the R2 key of the input picture")
    return key.strip()


def parse_input(raw: dict | None) -> JobSpec:
    if not isinstance(raw, dict):
        raise InputError("input must be an object")
    keys = raw.get("image_r2_keys")
    if keys is None:
        keys = [raw.get("image_r2_key")]
    if not isinstance(keys, (list, tuple)) or not keys:
        raise InputError("image_r2_keys must be a list of R2 keys of the input pictures")
    if len(keys) > MAX_IMAGES:
        raise InputError(f"at most {MAX_IMAGES} pictures")
    keys = tuple(_key(k) for k in keys)
    if len(set(keys)) != len(keys):
        raise InputError("the same picture was sent twice")

    together_steps = raw.get("together_steps")
    if together_steps is not None:
        together_steps = _int(together_steps, "together_steps")
        if not 0 <= together_steps <= MAX_TOGETHER_STEPS:
            raise InputError(f"together_steps must be between 0 and {MAX_TOGETHER_STEPS}")
    sampler_steps = raw.get("sampler_steps")
    if sampler_steps is not None:
        sampler_steps = _int(sampler_steps, "sampler_steps")
        if not MIN_SAMPLER_STEPS <= sampler_steps <= MAX_SAMPLER_STEPS:
            raise InputError(f"sampler_steps must be between {MIN_SAMPLER_STEPS} and {MAX_SAMPLER_STEPS}")

    quality = str(raw.get("quality") or "standard").strip().lower()
    if quality not in PIPELINE_TYPES:
        raise InputError(f"quality must be one of {', '.join(PIPELINE_TYPES)}")

    seed = _int(raw.get("seed", 42), "seed")
    if not 0 <= seed <= SEED_MAX:
        raise InputError(f"seed must be between 0 and {SEED_MAX}")

    texture_size = _int(raw.get("texture_size", 2048), "texture_size")
    if texture_size not in TEXTURE_SIZES:
        raise InputError(f"texture_size must be one of {', '.join(map(str, TEXTURE_SIZES))}")

    faces = _int(raw.get("decimation_target", 500_000), "decimation_target")
    if not MIN_FACES <= faces <= MAX_FACES:
        raise InputError(f"decimation_target must be between {MIN_FACES} and {MAX_FACES}")

    turntable = raw.get("turntable", True)
    if not isinstance(turntable, bool):
        raise InputError("turntable must be true or false")

    uid = raw.get("uid")
    if uid is not None and (not isinstance(uid, str) or not uid or "/" in uid or len(uid) > 128):
        raise InputError("uid must be a user id")

    return JobSpec(
        image_r2_keys=keys,
        together_steps=together_steps,
        sampler_steps=sampler_steps,
        pipeline_type=PIPELINE_TYPES[quality],
        quality=quality,
        seed=seed,
        texture_size=texture_size,
        decimation_target=faces,
        turntable=turntable,
        uid=uid,
    )


def output_key(leaf: str, job_id: str, uid: str | None) -> str:
    """Same layout as the ComfyUI workers (base/lib/r2.py)."""
    return f"users/{uid}/generations/{leaf}" if uid else f"{job_id}/{leaf}"
