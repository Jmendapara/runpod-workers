"""Unit tests for base/lib/inputs.py — r2_loras validation + download cache,
volume_inputs (CozyGPU volume mode) and the stale-input prune.

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


# ---- validate_input: volume_inputs (CozyGPU volume mode) ----

def test_no_volume_inputs_defaults_to_empty_list():
    data, err = inputs.validate_input(_valid_base())
    assert err is None
    assert data["volume_inputs"] == []


def test_valid_volume_inputs_pass_through():
    entries = [
        {"node_id": "1", "input_field": "image", "key": "cozygpu/inputs/abc123.png"},
        {"node_id": 1, "input_field": "video", "key": "cozygpu/inputs/def456.mp4"},
        {"node_id": "1", "input_field": "mask", "key": "cozygpu/inputs/x..y.png"},  # '..' inside a name is fine
    ]
    data, err = inputs.validate_input(_valid_base(volume_inputs=entries))
    assert err is None, err
    assert data["volume_inputs"] == entries


def test_volume_inputs_must_be_list():
    _expect_error(_valid_base(volume_inputs="nope"), "'volume_inputs' must be a list")
    _expect_error(_valid_base(volume_inputs={"node_id": "1"}), "'volume_inputs' must be a list")


def test_volume_inputs_entry_must_be_object():
    _expect_error(_valid_base(volume_inputs=["nope"]), "volume_inputs[0] must be an object")


def test_volume_inputs_requires_fields():
    _expect_error(
        _valid_base(volume_inputs=[{"input_field": "image", "key": "a.png"}]),
        "volume_inputs[0] missing required field 'node_id'",
    )
    _expect_error(
        _valid_base(volume_inputs=[{"node_id": "1", "key": "a.png"}]),
        "volume_inputs[0] missing required field 'input_field'",
    )
    _expect_error(
        _valid_base(volume_inputs=[{"node_id": "1", "input_field": "image"}]),
        "volume_inputs[0] missing required field 'key'",
    )


def test_volume_inputs_key_traversal_guards():
    bad_keys = (
        "/abs/x.png", "../x.png", "cozygpu/../../etc/passwd", "cozygpu/inputs/..",
        "", "a//b.png", "./x.png", "cozygpu/inputs/", "a\\b.png", 5, None,
    )
    for bad in bad_keys:
        _expect_error(
            _valid_base(volume_inputs=[{"node_id": "1", "input_field": "image", "key": bad}]),
            "volume_inputs[0] key must be a relative path inside the volume",
        )


def test_volume_inputs_duplicate_target_rejected_across_arrays():
    vol = {"node_id": "1", "input_field": "image", "key": "cozygpu/inputs/a.png"}
    r2 = {"node_id": 1, "input_field": "image", "r2_key": "users/u1/a.png"}
    img = {"node_id": "1", "input_field": "image", "name": "a.png", "image": "aGk="}
    _expect_error(
        _valid_base(r2_inputs=[r2], volume_inputs=[vol]),
        "volume_inputs[0] targets node_id=1, input_field=image",
    )
    _expect_error(
        _valid_base(volume_inputs=[vol], images=[img]),
        "images[0] targets node_id=1, input_field=image",
    )
    _expect_error(
        _valid_base(volume_inputs=[vol, dict(vol)]),
        "volume_inputs[1] targets node_id=1, input_field=image",
    )


def test_volume_inputs_distinct_fields_on_same_node_allowed():
    data, err = inputs.validate_input(_valid_base(
        r2_inputs=[{"node_id": "1", "input_field": "image", "r2_key": "users/u1/a.png"}],
        volume_inputs=[{"node_id": "1", "input_field": "mask", "key": "cozygpu/inputs/m.png"}],
    ))
    assert err is None, err


# ---- process_volume_inputs ----

class _VolumeEnv:
    """Temp dirs standing in for /runpod-volume and /comfyui/input, restored on exit.

    `mounted=False` models an endpoint with no network volume attached (the mount
    dir does not exist). ensure_audio_track is stubbed and records the paths it is
    handed, so tests can assert the r2_inputs post-step runs for volume inputs too.
    """

    def __init__(self, mounted=True):
        self.mounted = mounted

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.volume_dir = os.path.join(self.tmp.name, "volume")
        self.input_dir = os.path.join(self.tmp.name, "input")
        if self.mounted:
            os.makedirs(self.volume_dir)
        self.prev_volume = inputs.RUNPOD_VOLUME_DIR
        self.prev_input = inputs.COMFY_INPUT_DIR
        inputs.RUNPOD_VOLUME_DIR = self.volume_dir
        inputs.COMFY_INPUT_DIR = self.input_dir
        self.audio_fixed: list[str] = []
        fake_ffmpeg = types.ModuleType("lib.ffmpeg_helpers")
        fake_ffmpeg.ensure_audio_track = lambda path: self.audio_fixed.append(path) or False
        self.prev_ffmpeg = sys.modules.get("lib.ffmpeg_helpers")
        sys.modules["lib.ffmpeg_helpers"] = fake_ffmpeg
        return self

    def __exit__(self, *exc):
        inputs.RUNPOD_VOLUME_DIR = self.prev_volume
        inputs.COMFY_INPUT_DIR = self.prev_input
        if self.prev_ffmpeg is None:
            sys.modules.pop("lib.ffmpeg_helpers", None)
        else:
            sys.modules["lib.ffmpeg_helpers"] = self.prev_ffmpeg
        self.tmp.cleanup()
        return False

    def put(self, key, payload=b"x" * 16):
        path = os.path.join(self.volume_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)
        return path

    def local(self, filename):
        return os.path.join(self.input_dir, filename)


def _expect_value_error(fn, needle, exact=False):
    try:
        fn()
    except ValueError as exc:
        if exact:
            assert str(exc) == needle, f"expected {needle!r}, got {exc!r}"
        else:
            assert needle in str(exc), f"expected error containing {needle!r}, got {exc!r}"
    else:
        raise AssertionError(f"expected ValueError containing {needle!r}")


def test_process_volume_inputs_empty_list_is_noop_without_mount():
    with _VolumeEnv(mounted=False) as env:
        inputs.process_volume_inputs({"1": {"inputs": {}}}, [])
        assert not os.path.exists(env.input_dir)
        assert env.audio_fixed == []


def test_process_volume_inputs_requires_mount():
    with _VolumeEnv(mounted=False) as env:
        workflow = {"5": {"class_type": "LoadImage", "inputs": {"image": ""}}}
        _expect_value_error(
            lambda: inputs.process_volume_inputs(
                workflow, [{"node_id": "5", "input_field": "image", "key": "cozygpu/inputs/x.png"}]
            ),
            "No network volume mounted",
            exact=True,
        )
        assert workflow["5"]["inputs"]["image"] == ""
        assert not os.path.exists(env.input_dir)


def test_process_volume_inputs_missing_file():
    with _VolumeEnv() as env:
        env.put("cozygpu/inputs/other.png")
        workflow = {"5": {"class_type": "LoadImage", "inputs": {"image": ""}}}
        _expect_value_error(
            lambda: inputs.process_volume_inputs(
                workflow, [{"node_id": "5", "input_field": "image", "key": "cozygpu/inputs/missing.png"}]
            ),
            "Volume input not found: cozygpu/inputs/missing.png",
            exact=True,
        )
        assert workflow["5"]["inputs"]["image"] == ""
        assert not os.path.exists(env.local("missing.png"))
        assert env.audio_fixed == []


def test_process_volume_inputs_directory_key_is_not_a_file():
    with _VolumeEnv() as env:
        env.put("cozygpu/inputs/dir/inner.png")
        _expect_value_error(
            lambda: inputs.process_volume_inputs(
                {"5": {"inputs": {}}}, [{"node_id": "5", "input_field": "image", "key": "cozygpu/inputs/dir"}]
            ),
            "Volume input not found: cozygpu/inputs/dir",
            exact=True,
        )


def test_process_volume_inputs_copies_rewrites_and_repairs_audio():
    with _VolumeEnv() as env:
        png = b"\x89PNG\r\n\x1a\n" + b"0" * 32
        env.put("cozygpu/inputs/abc.png", png)
        env.put("cozygpu/inputs/def.mp4", b"\x00" * 64)
        workflow = {
            "5": {"class_type": "LoadImage", "inputs": {"image": ""}},
            "7": {"class_type": "VHS_LoadVideo", "inputs": {"video": ""}},
        }
        inputs.process_volume_inputs(workflow, [
            {"node_id": 5, "input_field": "image", "key": "cozygpu/inputs/abc.png"},
            {"node_id": "7", "input_field": "video", "key": "cozygpu/inputs/def.mp4"},
        ])
        # workflow rewritten to the bare basename (ComfyUI resolves it under /comfyui/input)
        assert workflow["5"]["inputs"]["image"] == "abc.png"
        assert workflow["7"]["inputs"]["video"] == "def.mp4"
        with open(env.local("abc.png"), "rb") as f:
            assert f.read() == png
        assert os.path.getsize(env.local("def.mp4")) == 64
        # same post-step as r2_inputs: ensure_audio_track runs on every copied file
        # (the real helper no-ops itself on non-video extensions)
        assert env.audio_fixed == [env.local("abc.png"), env.local("def.mp4")]
        # the source stays on the volume — the app owns its lifecycle
        assert os.path.exists(os.path.join(env.volume_dir, "cozygpu/inputs/abc.png"))


def test_process_volume_inputs_unknown_node_rejected():
    with _VolumeEnv() as env:
        env.put("cozygpu/inputs/abc.png")
        _expect_value_error(
            lambda: inputs.process_volume_inputs(
                {"5": {"inputs": {}}}, [{"node_id": "6", "input_field": "image", "key": "cozygpu/inputs/abc.png"}]
            ),
            "volume_inputs references node_id '6' not in the workflow",
        )
        assert not os.path.exists(env.local("abc.png"))


def test_process_volume_inputs_node_without_inputs_rejected():
    with _VolumeEnv() as env:
        env.put("cozygpu/inputs/abc.png")
        _expect_value_error(
            lambda: inputs.process_volume_inputs(
                {"5": {"class_type": "LoadImage"}}, [{"node_id": "5", "input_field": "image", "key": "cozygpu/inputs/abc.png"}]
            ),
            "Workflow node '5' has no 'inputs' dict",
        )


# ---- prune_stale_inputs ----

class _InputDir:
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prev = inputs.COMFY_INPUT_DIR
        inputs.COMFY_INPUT_DIR = self.tmp.name
        return self

    def __exit__(self, *exc):
        inputs.COMFY_INPUT_DIR = self.prev
        self.tmp.cleanup()
        return False

    def file(self, name, mtime):
        path = os.path.join(self.tmp.name, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x")
        os.utime(path, (mtime, mtime))
        return path


def test_prune_stale_inputs_removes_only_old_top_level_files():
    now = 1_000_000.0
    with _InputDir() as d:
        old = d.file("old.png", now - 2 * 3600)
        fresh = d.file("fresh.png", now - 60)
        nested = d.file("clipspace/ancient.png", now - 2 * 3600)
        os.utime(os.path.dirname(nested), (now - 2 * 3600, now - 2 * 3600))
        assert inputs.prune_stale_inputs(now=now) == 1
        assert not os.path.exists(old)
        assert os.path.exists(fresh)
        assert os.path.exists(nested)  # subfolders (ComfyUI's own) are left alone
        assert os.path.isdir(d.tmp.name)  # the dir itself survives
        assert inputs.prune_stale_inputs(now=now) == 0  # idempotent


def test_prune_stale_inputs_default_age_is_one_hour_and_boundary_is_strict():
    assert inputs.INPUT_MAX_AGE_SECONDS == 3600
    now = 1_000_000.0
    with _InputDir() as d:
        at_limit = d.file("at-limit.png", now - 3600)
        past_limit = d.file("past-limit.png", now - 3601)
        assert inputs.prune_stale_inputs(now=now) == 1
        assert os.path.exists(at_limit)
        assert not os.path.exists(past_limit)


def test_prune_stale_inputs_missing_dir_is_noop():
    prev = inputs.COMFY_INPUT_DIR
    inputs.COMFY_INPUT_DIR = os.path.join(tempfile.gettempdir(), "comfy-input-does-not-exist-" + os.urandom(4).hex())
    try:
        assert inputs.prune_stale_inputs() == 0
    finally:
        inputs.COMFY_INPUT_DIR = prev


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
