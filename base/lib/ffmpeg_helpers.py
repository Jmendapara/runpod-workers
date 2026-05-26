"""ffmpeg transforms used by output collectors and input preprocessing."""
from __future__ import annotations

import os
import subprocess
import tempfile


# Containers where AAC + h264 copy mux is reliable. Anything else (webm/mkv/avi)
# we leave alone and let the workflow surface its own error.
SILENT_MUX_COMPATIBLE_EXTS = {".mp4", ".mov", ".m4v"}


def ensure_audio_track(path: str) -> bool:
    """If `path` is a video with no audio stream, mux a silent stereo AAC track
    in-place. Returns True iff the file was modified.

    Used for workflows whose downstream nodes (e.g. VHS NormalizeAudioLoudness,
    LTX audio VAE) assume an audio stream exists. Adding silence lets the
    workflow execute; the muxed track adds ~1-3 KB and no audible content.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SILENT_MUX_COMPATIBLE_EXTS:
        return False

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "a",
             "-show_entries", "stream=index",
             "-of", "csv=p=0", path],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"worker-comfyui - ffprobe failed for {path}: {exc}", flush=True)
        return False

    if probe.returncode == 0 and probe.stdout.strip():
        return False  # already has at least one audio stream

    tmp_path = f"{path}.silenced{ext}"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", path,
             "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
             "-c:v", "copy", "-c:a", "aac",
             "-shortest",
             "-map", "0:v:0", "-map", "1:a:0",
             tmp_path],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"worker-comfyui - silent-mux failed for {path}: {stderr}", flush=True)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        os.replace(tmp_path, path)
        print(f"worker-comfyui - Muxed silent audio track into {path} (input had no audio)", flush=True)
        return True
    except Exception as exc:
        print(f"worker-comfyui - silent-mux error for {path}: {exc}", flush=True)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def flac_to_wav(file_bytes: bytes, filename: str) -> tuple[bytes, str]:
    """Re-encode FLAC bytes to stereo 44.1 kHz WAV. Returns (bytes, new_filename).

    On failure returns the original bytes unchanged.
    """
    flac_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            tmp.write(file_bytes)
            flac_path = tmp.name

        wav_path = flac_path.replace(".flac", ".wav")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", flac_path,
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                wav_path,
            ],
            capture_output=True, timeout=60,
        )
        os.remove(flac_path)
        flac_path = None

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"worker-comfyui - ffmpeg FLAC→WAV failed: {stderr}", flush=True)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
            return file_bytes, filename

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        os.remove(wav_path)
        new_filename = os.path.splitext(filename)[0] + ".wav"
        print(
            f"worker-comfyui - Converted {filename} FLAC→WAV "
            f"({len(file_bytes)} → {len(wav_bytes)} bytes)",
            flush=True,
        )
        return wav_bytes, new_filename
    except Exception as exc:
        print(f"worker-comfyui - FLAC→WAV error: {exc}", flush=True)
        for p in (flac_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return file_bytes, filename
