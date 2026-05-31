"""Output collectors — map ComfyUI's output keys to the response shape.

Driven by `output.type` in model.yaml. Pluggable transform (FLAC→WAV) and
filter (VHS sidecar) attach based on per-model flags.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import OutputConfig
from .ffmpeg_helpers import flac_to_wav, extract_poster, make_preview_clip


# Type aliases
TransformFn = Callable[[bytes, str], tuple[bytes, str]]
FilterFn = Callable[[str, str, bool], bool]  # (comfy_key, filename, has_audio_variant) -> keep?


def _vhs_audio_preferred(comfy_key: str, filename: str, has_audio_variant: bool) -> bool:
    """Prefer *-audio.mp4 over silent .mp4 / .png thumbnail.

    When the driving video has no audio track, VHS_VideoCombine only writes the
    silent .mp4 — accept it as fallback.
    """
    if not filename:
        return False
    if filename.endswith("-audio.mp4"):
        return True
    if not has_audio_variant and filename.endswith(".mp4"):
        return True
    # Drop silent mp4 + png thumbnail when audio variant exists
    return False


@dataclass
class Collector:
    comfy_key: str           # key in ComfyUI's history outputs ("images" | "videos" | "audio" | "gifs")
    result_key: str          # key in the final response ("images" | "videos" | "audio")
    transform: Optional[TransformFn] = None
    filter: Optional[FilterFn] = None


OUTPUT_TYPE_MAP: dict[str, list[Collector]] = {
    "image": [
        Collector("images", "images"),
    ],
    "video": [
        Collector("images", "images"),
        Collector("videos", "videos"),
    ],
    "audio": [
        Collector("audio", "audio"),
    ],
    "image+audio": [
        Collector("images", "images"),
        Collector("audio", "audio"),
    ],
    "image+video": [
        Collector("images", "images"),
        Collector("videos", "videos"),
    ],
    "image+video+gifs": [
        Collector("images", "images"),
        Collector("videos", "videos"),
        Collector("gifs", "videos"),  # VHS emits MP4 under "gifs"
    ],
}


@dataclass
class CollectorSet:
    collectors: list[Collector]
    vhs_sidecar_filter: bool = False

    def _collected_keys(self) -> set[str]:
        return {c.comfy_key for c in self.collectors}

    def harvest(
        self,
        history: dict,
        comfy_client,
        job_id: str,
        uploader,
        uid: str | None = None,
    ) -> dict:
        """Walk ComfyUI history outputs and produce the final response shape.

        Returns dict with keys like "images", "videos", "audio", and "errors".
        Honors R2 upload if uploader != None, else returns base64.
        """
        result: dict[str, list] = {}
        errors: list[str] = []

        outputs = history.get("outputs", {})
        if not outputs:
            errors.append("No outputs found in ComfyUI history")

        has_audio_variant = False
        if self.vhs_sidecar_filter:
            for _nid, _nout in outputs.items():
                for _mk in ("images", "videos", "gifs"):
                    for _item in _nout.get(_mk, []):
                        if (_item.get("filename") or "").endswith("-audio.mp4"):
                            has_audio_variant = True
                            break
                    if has_audio_variant:
                        break
                if has_audio_variant:
                    break
            if not has_audio_variant:
                print(
                    "worker-comfyui - No -audio.mp4 found in outputs, will accept silent .mp4",
                    flush=True,
                )

        handled_keys = self._collected_keys()

        for node_id, node_output in outputs.items():
            for collector in self.collectors:
                items = node_output.get(collector.comfy_key) or []
                if not items:
                    continue
                print(
                    f"worker-comfyui - Node {node_id}: {len(items)} {collector.comfy_key}",
                    flush=True,
                )
                for item_info in items:
                    filename = item_info.get("filename")
                    subfolder = item_info.get("subfolder", "")
                    item_type = item_info.get("type")

                    if item_type == "temp":
                        continue
                    if not filename:
                        errors.append(
                            f"Skipping {collector.comfy_key} in node {node_id}: missing filename"
                        )
                        continue

                    if collector.filter and not collector.filter(
                        collector.comfy_key, filename, has_audio_variant
                    ):
                        print(f"worker-comfyui - Skipping sidecar {filename}", flush=True)
                        continue

                    file_bytes = comfy_client.fetch_view(filename, subfolder, item_type)
                    if not file_bytes:
                        errors.append(
                            f"Failed to fetch {collector.comfy_key} data for {filename}"
                        )
                        continue

                    if collector.transform:
                        file_bytes, filename = collector.transform(file_bytes, filename)

                    item = _to_response_item(file_bytes, filename, job_id, uid, uploader, errors)
                    if item is not None:
                        # Generate a poster + preview clip for any VIDEO file, regardless of which
                        # ComfyUI output key it arrived under. Native SaveVideo reports its MP4 under
                        # "images" (result_key "images"), not "videos" — so gating on result_key alone
                        # misses it. Gate on the file actually being a video instead.
                        is_video_file = (filename or "").lower().endswith((".mp4", ".webm", ".mov", ".mkv"))
                        if (collector.result_key == "videos" or is_video_file) and uploader is not None:
                            _attach_video_derivatives(item, file_bytes, filename, job_id, uid, uploader)
                        result.setdefault(collector.result_key, []).append(item)

            other_keys = [k for k in node_output.keys() if k not in handled_keys]
            if other_keys:
                print(
                    f"worker-comfyui - WARNING: Node {node_id} produced unhandled keys: {other_keys}",
                    flush=True,
                )

        if errors:
            result["errors"] = errors
        return result


def _attach_video_derivatives(item, file_bytes, filename, job_id, uid, uploader) -> None:
    """Generate a poster frame + low-res preview clip and attach their R2 keys to `item`.

    `posterKey` / `previewKey` are stored by the app and signed on read. Best-effort:
    any failure just leaves the key off and the app falls back to the full video.
    """
    ext = os.path.splitext(filename)[1] or ".mp4"
    try:
        poster = extract_poster(file_bytes, ext)
        if poster:
            key, _url = uploader.upload_returning_key(poster, "poster.webp", job_id, uid=uid)
            item["posterKey"] = key
            print(f"worker-comfyui - Uploaded poster for {filename}", flush=True)
    except Exception as exc:
        print(f"worker-comfyui - poster generation error for {filename}: {exc}", flush=True)
    try:
        preview = make_preview_clip(file_bytes, ext)
        if preview:
            key, _url = uploader.upload_returning_key(preview, "preview.mp4", job_id, uid=uid)
            item["previewKey"] = key
            print(f"worker-comfyui - Uploaded preview clip for {filename}", flush=True)
    except Exception as exc:
        print(f"worker-comfyui - preview generation error for {filename}: {exc}", flush=True)


def _to_response_item(
    file_bytes: bytes,
    filename: str,
    job_id: str,
    uid: str | None,
    uploader,
    errors: list[str],
) -> dict | None:
    if uploader is not None:
        try:
            s3_url = uploader.upload(file_bytes, filename, job_id, uid=uid)
            print(f"worker-comfyui - Uploaded {filename} to R2", flush=True)
            return {"filename": filename, "type": "s3_url", "data": s3_url}
        except Exception as exc:
            errors.append(f"Error uploading {filename} to R2: {exc}")
            return None

    try:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > 15:
            print(
                f"worker-comfyui - WARNING: {filename} is {file_size_mb:.1f} MB. "
                f"Large responses may be truncated by RunPod. Configure R2 upload for reliability.",
                flush=True,
            )
        return {
            "filename": filename,
            "type": "base64",
            "data": base64.b64encode(file_bytes).decode("utf-8"),
        }
    except Exception as exc:
        errors.append(f"Error encoding {filename} to base64: {exc}")
        return None


def build_collectors(output_cfg: OutputConfig) -> CollectorSet:
    if output_cfg.type not in OUTPUT_TYPE_MAP:
        raise ValueError(f"Unknown output.type: {output_cfg.type}")
    cols = [
        Collector(c.comfy_key, c.result_key, c.transform, c.filter)
        for c in OUTPUT_TYPE_MAP[output_cfg.type]
    ]

    if output_cfg.transcode_flac_to_wav:
        for c in cols:
            if c.comfy_key == "audio":
                c.transform = flac_to_wav

    if output_cfg.vhs_sidecar_filter:
        for c in cols:
            if c.comfy_key in ("videos", "gifs"):
                c.filter = _vhs_audio_preferred

    return CollectorSet(collectors=cols, vhs_sidecar_filter=output_cfg.vhs_sidecar_filter)
