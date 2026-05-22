"""ffmpeg transforms used by output collectors."""
from __future__ import annotations

import os
import subprocess
import tempfile


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
