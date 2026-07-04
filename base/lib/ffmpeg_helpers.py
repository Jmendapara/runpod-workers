"""ffmpeg transforms used by output collectors and input preprocessing."""
from __future__ import annotations

import os
import subprocess
import tempfile


# Containers where we can copy the video stream and (re)mux/repair the audio
# track. aac muxes reliably into mp4/mov/m4v/mkv; webm needs opus. Anything else
# (avi, etc.) we leave alone and let the workflow surface its own error.
SILENT_MUX_COMPATIBLE_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}

# Pad audio when it ends more than this many seconds before the video does. The
# LTX Extend Video workflow slices the final ~refSeconds of the input clip's
# audio (TrimAudioDuration "Get end audio"); if the audio ends before that
# window, the node raises "Start time must be less than end time and be within
# the audio length." Filling to the video length with trailing silence keeps
# that slice valid. The tolerance absorbs the sub-frame audio/video duration
# skew present in almost every container without triggering a needless re-encode.
AUDIO_PAD_TOLERANCE_SEC = 0.1


def _audio_codec_for_ext(ext: str) -> str:
    """Audio codec to use when (re)muxing into the given container extension."""
    return "libopus" if ext == ".webm" else "aac"


def _ffprobe_duration(path: str, select: str | None) -> float | None:
    """Return a duration in seconds via ffprobe, or None if unavailable.

    select=None reads the container (format) duration; select="v:0"/"a:0" reads
    a specific stream's duration. Stream-level duration is N/A in some containers
    (notably webm), so callers should fall back to the format duration.
    """
    if select is None:
        entries = "format=duration"
        sel_args = []
    else:
        entries = "stream=duration"
        sel_args = ["-select_streams", select]
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", *sel_args,
             "-show_entries", entries, "-of", "csv=p=0", path],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"worker-comfyui - ffprobe duration failed for {path}: {exc}", flush=True)
        return None
    if probe.returncode != 0:
        return None
    try:
        return float(probe.stdout.decode("utf-8", errors="replace").strip())
    except (TypeError, ValueError):
        return None


def ensure_audio_track(path: str) -> bool:
    """Guarantee `path` carries an audio stream that spans the full video.

    Repairs two cases in-place, returning True iff the file was modified:
      * No audio stream at all  → mux a silent stereo track length-matched to
        the video.
      * Audio shorter than the video → pad the existing audio with trailing
        silence up to the video length.

    Downstream nodes in the LTX Extend Video workflow (VHS NormalizeAudioLoudness,
    TrimAudioDuration "Get end audio", LTX audio VAE) assume the input clip's
    audio reaches the end of the video. Missing or short audio otherwise fails
    audio trimming with "Start time must be less than end time...". The injected
    silence is inaudible and adds only a few KB.
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
    has_audio = probe.returncode == 0 and bool(probe.stdout.strip())

    codec = _audio_codec_for_ext(ext)
    tmp_path = f"{path}.audiofix{ext}"

    if has_audio:
        video_dur = _ffprobe_duration(path, "v:0") or _ffprobe_duration(path, None)
        if video_dur is None:
            return False  # can't establish the target length — leave it alone
        audio_dur = _ffprobe_duration(path, "a:0")
        # audio_dur is None when the container doesn't expose a stream duration
        # (e.g. webm); pad defensively in that case since we can't prove it's
        # long enough. When known and already spanning the video, do nothing.
        if audio_dur is not None and audio_dur >= video_dur - AUDIO_PAD_TOLERANCE_SEC:
            return False
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-map", "0:v:0", "-map", "0:a:0",
            "-af", "apad",          # pad the existing audio with trailing silence
            "-c:v", "copy", "-c:a", codec,
            "-shortest",            # apad is infinite → output ends with the video
            tmp_path,
        ]
        action = (
            f"Padded short audio track to video length "
            f"(audio={'unknown' if audio_dur is None else f'{audio_dur:.2f}s'}, "
            f"video={video_dur:.2f}s)"
        )
        fail_label = "audio-pad"
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "copy", "-c:a", codec,
            "-shortest",
            "-map", "0:v:0", "-map", "1:a:0",
            tmp_path,
        ]
        action = "Muxed silent audio track (input had no audio)"
        fail_label = "silent-mux"

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"worker-comfyui - {fail_label} failed for {path}: {stderr}", flush=True)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        os.replace(tmp_path, path)
        print(f"worker-comfyui - {action}: {path}", flush=True)
        return True
    except Exception as exc:
        print(f"worker-comfyui - {fail_label} error for {path}: {exc}", flush=True)
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
