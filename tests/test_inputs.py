"""Unit tests for base/lib/inputs.py — r2_loras validation + download cache.

CI only validates model.yaml schemas, so run these locally:
    python3 tests/test_inputs.py
(also collectable by pytest if installed). No boto3 required — lib.r2 is
stubbed via sys.modules before lib.inputs ever resolves it.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base")
sys.path.insert(0, BASE_DIR)


class StubTarget:
    """Stand-in for lib.r2.BucketTarget — only the fields inputs.py reads."""

    def __init__(self, input_bucket: str = "bucket"):
        self.input_bucket = input_bucket


class StubS3:
    def __init__(self, target=None, payload: bytes = b"x" * 16):
        self.target = target
        self.payload = payload
        self.downloads: list[tuple[str, str, str]] = []

    def download_file(self, bucket, key, local_path):
        self.downloads.append((bucket, key, local_path))
        with open(local_path, "wb") as f:
            f.write(self.payload)


_fake_r2 = types.ModuleType("lib.r2")
_fake_r2.make_s3_client = StubS3  # each call returns a fresh stub; tests swap this
sys.modules["lib.r2"] = _fake_r2

from lib import inputs  # noqa: E402


def _valid_base(**extra):
    return {"workflow": {"1": {"class_type": "KSampler", "inputs": {}}}, **extra}


def _expect_error(job_input, needle):
    data, err = inputs.validate_input(job_input)
    assert data is None, f"expected rejection, got {data}"
    assert err and needle in err, f"expected error containing {needle!r}, got {err!r}"


# ---- validate_input: r2_loras ----

def test_no_r2_loras_defaults_to_empty_list():
    data, err = inputs.validate_input(_valid_base())
    assert err is None
    assert data["r2_loras"] == []


def test_valid_r2_loras_pass_through():
    entries = [
        {"r2_key": "users/u1/loras/abc.safetensors", "filename": "abc.safetensors", "size_bytes": 123},
        {"r2_key": "users/u1/loras/def.safetensors", "filename": "def.safetensors"},
    ]
    data, err = inputs.validate_input(_valid_base(r2_loras=entries))
    assert err is None, err
    assert data["r2_loras"] == entries


def test_r2_loras_must_be_list():
    _expect_error(_valid_base(r2_loras="nope"), "'r2_loras' must be a list")


def test_r2_loras_entry_must_be_object():
    _expect_error(_valid_base(r2_loras=["nope"]), "must be an object")


def test_r2_loras_requires_string_fields():
    _expect_error(_valid_base(r2_loras=[{"filename": "a.safetensors"}]), "r2_key")
    _expect_error(_valid_base(r2_loras=[{"r2_key": "k"}]), "filename")
    _expect_error(_valid_base(r2_loras=[{"r2_key": "", "filename": "a.safetensors"}]), "r2_key")
    _expect_error(_valid_base(r2_loras=[{"r2_key": 5, "filename": "a.safetensors"}]), "r2_key")


def test_r2_loras_filename_traversal_and_suffix_guards():
    for bad in ("../evil.safetensors", "sub/x.safetensors", "/abs.safetensors", "x.bin", ".safetensors"):
        _expect_error(
            _valid_base(r2_loras=[{"r2_key": "k", "filename": bad}]),
            "bare .safetensors basename",
        )


def test_r2_loras_size_bytes_must_be_positive_int():
    for bad in (0, -1, "5", True, 1.5):
        _expect_error(
            _valid_base(r2_loras=[{"r2_key": "k", "filename": "a.safetensors", "size_bytes": bad}]),
            "size_bytes",
        )


def test_r2_loras_conflicting_duplicate_filenames_rejected():
    entries = [
        {"r2_key": "users/u1/loras/a.safetensors", "filename": "a.safetensors"},
        {"r2_key": "users/u2/loras/other.safetensors", "filename": "a.safetensors"},
    ]
    _expect_error(_valid_base(r2_loras=entries), "conflicting r2_keys")


def test_r2_loras_same_key_duplicate_allowed():
    entry = {"r2_key": "users/u1/loras/a.safetensors", "filename": "a.safetensors"}
    data, err = inputs.validate_input(_valid_base(r2_loras=[entry, dict(entry)]))
    assert err is None, err


# ---- process_r2_loras ----

class _LoraEnv:
    """Temp loras dir + a resolved bucket target + fresh stub client, restored on exit.

    The handler resolves the job's BucketTarget (lib.r2) and hands it to
    process_r2_loras / process_r2_inputs; `bucket=None` models an endpoint with
    no R2 configured (target None).
    """

    def __init__(self, bucket="test-bucket"):
        self.bucket = bucket
        self.target = None if bucket is None else StubTarget(input_bucket=bucket)

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prev_dir = inputs.COMFY_LORA_DIR
        inputs.COMFY_LORA_DIR = self.tmp.name
        self.stub = StubS3()
        _fake_r2.make_s3_client = lambda target: self.stub
        return self

    def __exit__(self, *exc):
        inputs.COMFY_LORA_DIR = self.prev_dir
        self.tmp.cleanup()
        return False

    def path(self, filename):
        return os.path.join(self.tmp.name, filename)


def test_process_r2_loras_downloads_fresh_file():
    with _LoraEnv() as env:
        inputs.process_r2_loras([{ "r2_key": "users/u1/loras/a.safetensors", "filename": "a.safetensors", "size_bytes": 16 }], env.target)
        assert len(env.stub.downloads) == 1
        assert env.stub.downloads[0][0] == "test-bucket"
        assert os.path.getsize(env.path("a.safetensors")) == 16
        # atomic rename: no .part temp files left behind
        assert [f for f in os.listdir(env.tmp.name) if ".part-" in f] == []


def test_process_r2_loras_cache_hit_skips_download():
    with _LoraEnv() as env:
        with open(env.path("a.safetensors"), "wb") as f:
            f.write(b"x" * 16)
        inputs.process_r2_loras([{ "r2_key": "k", "filename": "a.safetensors", "size_bytes": 16 }], env.target)
        assert env.stub.downloads == []


def test_process_r2_loras_cache_hit_without_size_hint():
    with _LoraEnv() as env:
        with open(env.path("a.safetensors"), "wb") as f:
            f.write(b"x" * 5)
        inputs.process_r2_loras([{ "r2_key": "k", "filename": "a.safetensors" }], env.target)
        assert env.stub.downloads == []


def test_process_r2_loras_size_mismatch_redownloads():
    with _LoraEnv() as env:
        with open(env.path("a.safetensors"), "wb") as f:
            f.write(b"x" * 5)  # truncated leftover
        inputs.process_r2_loras([{ "r2_key": "k", "filename": "a.safetensors", "size_bytes": 16 }], env.target)
        assert len(env.stub.downloads) == 1
        assert os.path.getsize(env.path("a.safetensors")) == 16


def test_process_r2_loras_requires_bucket_target():
    with _LoraEnv(bucket=None) as env:
        try:
            inputs.process_r2_loras([{ "r2_key": "k", "filename": "a.safetensors" }], env.target)
        except ValueError as exc:
            assert "No input bucket configured" in str(exc)
        else:
            raise AssertionError("expected ValueError when the job has no bucket target")
        assert env.stub.downloads == []


def test_process_r2_loras_empty_list_is_noop():
    # Must not require a bucket target or touch disk.
    inputs.process_r2_loras([], None)


def test_process_r2_loras_downloads_from_target_input_bucket():
    with _LoraEnv(bucket="uploads-bucket") as env:
        inputs.process_r2_loras([{ "r2_key": "k", "filename": "a.safetensors" }], env.target)
        assert env.stub.downloads[0][0] == "uploads-bucket"


# ---- validate_input: bucket_profile ----

def test_bucket_profile_absent_is_none():
    data, err = inputs.validate_input(_valid_base())
    assert err is None
    assert data["bucket_profile"] is None


def test_bucket_profile_is_trimmed_and_passed_through():
    data, err = inputs.validate_input(_valid_base(bucket_profile="  pd-qa "))
    assert err is None
    assert data["bucket_profile"] == "pd-qa"


def test_bucket_profile_rejects_non_string_or_blank():
    _expect_error(_valid_base(bucket_profile=""), "'bucket_profile' must be a non-empty string")
    _expect_error(_valid_base(bucket_profile="   "), "'bucket_profile' must be a non-empty string")
    _expect_error(_valid_base(bucket_profile=7), "'bucket_profile' must be a non-empty string")
    _expect_error(_valid_base(bucket_profile={"name": "pd-qa"}), "'bucket_profile' must be a non-empty string")


# ---- process_r2_inputs: input bucket comes from the target ----

def test_process_r2_inputs_requires_bucket_target():
    workflow = {"5": {"class_type": "LoadImage", "inputs": {"image": ""}}}
    try:
        inputs.process_r2_inputs(workflow, [{"node_id": "5", "input_field": "image", "r2_key": "users/u1/uploads/x.png"}], None)
    except ValueError as exc:
        assert "No input bucket configured" in str(exc)
    else:
        raise AssertionError("expected ValueError when the job has no bucket target")


def test_process_r2_inputs_downloads_from_target_input_bucket_and_rewrites_workflow():
    stub = StubS3(payload=b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    _fake_r2.make_s3_client = lambda target: stub
    fake_ffmpeg = types.ModuleType("lib.ffmpeg_helpers")
    fake_ffmpeg.ensure_audio_track = lambda path: None
    sys.modules["lib.ffmpeg_helpers"] = fake_ffmpeg
    tmp = tempfile.TemporaryDirectory()
    prev_dir = inputs.COMFY_INPUT_DIR
    inputs.COMFY_INPUT_DIR = tmp.name
    try:
        workflow = {"5": {"class_type": "LoadImage", "inputs": {"image": ""}}}
        inputs.process_r2_inputs(
            workflow,
            [{"node_id": "5", "input_field": "image", "r2_key": "users/u1/uploads/x.png"}],
            StubTarget(input_bucket="osg-uploads"),
        )
        assert stub.downloads == [("osg-uploads", "users/u1/uploads/x.png", os.path.join(tmp.name, "x.png"))]
        assert workflow["5"]["inputs"]["image"] == "x.png"
    finally:
        inputs.COMFY_INPUT_DIR = prev_dir
        tmp.cleanup()


if __name__ == "__main__":
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
