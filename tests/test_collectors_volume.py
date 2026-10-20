"""Unit tests for base/lib/collectors.py — CozyGPU volume-mode output return.

With no bucket target, outputs larger than VOLUME_RETURN_THRESHOLD_BYTES are
written to the network volume (when the app's marker file is present) and
returned as {"type": "volume", ...}; everything else stays base64.

CI only validates model.yaml schemas, so run these locally:
    python3 tests/test_collectors_volume.py
(also collectable by pytest if installed). No PyYAML / jsonschema needed: lib.config
is stubbed when it cannot be imported. ffmpeg is never invoked (image outputs only).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base")
sys.path.insert(0, BASE_DIR)

try:
    from lib.config import OutputConfig  # needs PyYAML + jsonschema
except ImportError:
    @dataclass(frozen=True)
    class OutputConfig:  # mirrors lib.config.OutputConfig — only the fields collectors reads
        type: str
        vhs_sidecar_filter: bool = False
        transcode_flac_to_wav: bool = False

    _fake_config = types.ModuleType("lib.config")
    _fake_config.OutputConfig = OutputConfig
    sys.modules["lib.config"] = _fake_config

# test_inputs.py stubs lib.ffmpeg_helpers (ensure_audio_track only) in sys.modules and
# leaves it there; collectors needs the real module's flac_to_wav (stdlib-only, safe to
# import). Drop a leaked stub so both files can share one pytest process in any order.
_leaked = sys.modules.get("lib.ffmpeg_helpers")
if _leaked is not None and not hasattr(_leaked, "flac_to_wav"):
    del sys.modules["lib.ffmpeg_helpers"]

from lib import collectors  # noqa: E402

THRESHOLD = collectors.VOLUME_RETURN_THRESHOLD_BYTES
SMALL = b"s" * THRESHOLD        # exactly AT the threshold: must stay inline
LARGE = b"L" * (THRESHOLD + 1)  # one byte over: goes to the volume

MARKER = {
    "volumeId": "vol_abc123",
    "dataCenter": "US-KS-2",
    "app": "cozygpu",
    "createdAt": "2026-10-20T00:00:00Z",
}


class FakeComfy:
    """Stand-in for lib.comfy_client.ComfyClient — only fetch_view is used by harvest."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def fetch_view(self, filename, subfolder, file_type):
        return self.files.get(filename)


class FakeUploader:
    def __init__(self):
        self.uploaded: list[tuple] = []

    def upload(self, file_bytes, filename, job_id, uid=None):
        self.uploaded.append((filename, len(file_bytes), job_id, uid))
        return f"https://r2.example/{job_id}/{filename}?sig"


def _history(*filenames):
    return {"outputs": {"9": {"images": [
        {"filename": f, "subfolder": "", "type": "output"} for f in filenames
    ]}}}


class _Volume:
    """Temp dir standing in for /runpod-volume, restored on exit.

    marker=<dict> writes it as JSON; raw=<str> writes the text verbatim (for the
    malformed cases); neither → no marker file at all.
    """

    def __init__(self, marker=None, raw=None):
        self.marker = marker
        self.raw = raw

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.prev = collectors.RUNPOD_VOLUME_DIR
        collectors.RUNPOD_VOLUME_DIR = self.dir
        text = self.raw if self.raw is not None else (json.dumps(self.marker) if self.marker is not None else None)
        if text is not None:
            with open(self.path(collectors.VOLUME_MARKER_FILENAME), "w", encoding="utf-8") as f:
                f.write(text)
        return self

    def __exit__(self, *exc):
        collectors.RUNPOD_VOLUME_DIR = self.prev
        self.tmp.cleanup()
        return False

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def outputs_root(self):
        return self.path("cozygpu")


def _harvest(files: dict[str, bytes], uploader=None, job_id="job-1"):
    cs = collectors.build_collectors(OutputConfig(type="image"))
    return cs.harvest(_history(*files), comfy_client=FakeComfy(files), job_id=job_id, uploader=uploader)


# ---- the four required cases ----

def test_no_marker_large_file_stays_base64():
    with _Volume() as vol:
        out = _harvest({"big.png": LARGE})
        [item] = out["images"]
        assert item["type"] == "base64"
        assert base64.b64decode(item["data"]) == LARGE
        assert "errors" not in out
        assert not os.path.exists(vol.outputs_root())


def test_marker_small_file_stays_base64():
    with _Volume(marker=MARKER) as vol:
        out = _harvest({"small.png": SMALL})
        [item] = out["images"]
        assert item["type"] == "base64"
        assert base64.b64decode(item["data"]) == SMALL
        assert "errors" not in out
        assert not os.path.exists(vol.outputs_root())


def test_marker_large_file_returns_volume_item_and_copies_file():
    with _Volume(marker=MARKER) as vol:
        out = _harvest({"big.png": LARGE}, job_id="job-42")
        [item] = out["images"]
        assert item == {
            "filename": "big.png",
            "type": "volume",
            "volume_id": "vol_abc123",
            "data_center": "US-KS-2",
            "key": "cozygpu/outputs/job-42/big.png",
            "size_bytes": THRESHOLD + 1,
        }
        # the key is relative to the mount and the file is byte-identical
        with open(vol.path(*item["key"].split("/")), "rb") as f:
            assert f.read() == LARGE
        assert os.listdir(vol.path("cozygpu", "outputs", "job-42")) == ["big.png"]  # no .part leftovers
        assert "errors" not in out


def test_malformed_marker_stays_base64():
    bad_markers = (
        "{not json",
        "",
        "[]",
        "null",
        '"vol_abc123"',
        json.dumps({"dataCenter": "US-KS-2"}),                       # no volumeId
        json.dumps({"volumeId": "vol_abc123"}),                      # no dataCenter
        json.dumps({"volumeId": "", "dataCenter": "US-KS-2"}),       # blank id
        json.dumps({"volumeId": 5, "dataCenter": "US-KS-2"}),        # wrong type
        json.dumps({"volumeId": "vol_abc123", "dataCenter": None}),
    )
    for raw in bad_markers:
        with _Volume(raw=raw) as vol:
            out = _harvest({"big.png": LARGE})
            [item] = out["images"]
            assert item["type"] == "base64", raw
            assert base64.b64decode(item["data"]) == LARGE, raw
            assert "errors" not in out, raw
            assert not os.path.exists(vol.outputs_root()), raw


# ---- edges around the required cases ----

def test_load_volume_marker_shape():
    with _Volume(marker=MARKER):
        assert collectors.load_volume_marker() == {"volume_id": "vol_abc123", "data_center": "US-KS-2"}
    with _Volume(marker={"volumeId": "  vol_x ", "dataCenter": " EU-RO-1 "}):
        assert collectors.load_volume_marker() == {"volume_id": "vol_x", "data_center": "EU-RO-1"}
    with _Volume():
        assert collectors.load_volume_marker() is None


def test_mixed_sizes_split_per_file():
    with _Volume(marker=MARKER) as vol:
        out = _harvest({"a.png": SMALL, "b.png": LARGE})
        assert {i["filename"]: i["type"] for i in out["images"]} == {"a.png": "base64", "b.png": "volume"}
        assert os.listdir(vol.path("cozygpu", "outputs", "job-1")) == ["b.png"]


def test_marker_parsed_once_per_job():
    calls: list[int] = []
    orig = collectors.load_volume_marker
    collectors.load_volume_marker = lambda: calls.append(1) or orig()
    try:
        with _Volume(marker=MARKER):
            out = _harvest({"a.png": LARGE, "b.png": LARGE, "c.png": SMALL})
            assert [i["type"] for i in out["images"]] == ["volume", "volume", "base64"]
    finally:
        collectors.load_volume_marker = orig
    assert calls == [1]


def test_bucket_mode_is_unchanged_and_ignores_marker():
    with _Volume(marker=MARKER) as vol:
        up = FakeUploader()
        out = _harvest({"big.png": LARGE}, uploader=up)
        [item] = out["images"]
        assert item == {"filename": "big.png", "type": "s3_url", "data": "https://r2.example/job-1/big.png?sig"}
        assert up.uploaded == [("big.png", THRESHOLD + 1, "job-1", None)]
        assert not os.path.exists(vol.outputs_root())


def test_unmounted_volume_stays_base64():
    with _Volume() as vol:
        collectors.RUNPOD_VOLUME_DIR = vol.path("not-mounted")
        out = _harvest({"big.png": LARGE})
        assert out["images"][0]["type"] == "base64"


def test_volume_write_failure_falls_back_to_base64():
    with _Volume(marker=MARKER) as vol:
        # Make the outputs root a FILE so the directory cannot be created.
        with open(vol.outputs_root(), "wb") as f:
            f.write(b"")
        out = _harvest({"big.png": LARGE})
        [item] = out["images"]
        assert item["type"] == "base64"
        assert base64.b64decode(item["data"]) == LARGE
        assert "errors" not in out


def test_output_filename_is_reduced_to_a_basename():
    with _Volume(marker=MARKER) as vol:
        out = _harvest({"sub/dir/big.png": LARGE})
        [item] = out["images"]
        assert item["type"] == "volume"
        assert item["filename"] == "big.png"
        assert item["key"] == "cozygpu/outputs/job-1/big.png"
        assert os.path.isfile(vol.path("cozygpu", "outputs", "job-1", "big.png"))


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
