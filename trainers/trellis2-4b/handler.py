"""RunPod serverless handler: one picture → a textured, PBR 3D model with TRELLIS.2-4B.

Per job:
  1. download the picture from R2 (input contract: worker_config.py) — or up to four pictures of the
     same object from different sides
  2. TRELLIS.2 image-to-3D, the official pipeline and sampler settings (pipeline.json); the
     background is removed with BiRefNet unless the picture already has transparency. Several
     pictures steer the same samplers together (multi_image.py)
  3. bake the GLB exactly as the official demo does (o_voxel.postprocess.to_glb: remesh, UV
     unwrap, base colour + metallic/roughness textures, WebP)
  4. render a 360° turntable of the model under a studio light (the app's grid preview)
  5. upload both to R2 and return them as { type: 's3_url' } items

Output:
  { "models": [{ type, data, r2_key, filename, format: 'glb', bytes, faces, vertices, texture_size }],
    "videos": [{ type, data, r2_key, filename, width, height, fps, frames }],
    "seed": int, "timings": { stage: seconds }, "gpu": str,
    "pictures": int, and with several pictures "together_steps": int, "generate_timings": { stage: seconds }
    and "input_views": [{ theta, phi, match }] — where each picture is seen on the model }

R2 is required (a GLB is tens of MB): BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID,
BUCKET_SECRET_ACCESS_KEY, R2_BUCKET_NAME (+ optional R2_INPUT_BUCKET_NAME).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod  # noqa: E402
from runpod.serverless.modules.rp_progress import _thread_target as send_progress  # noqa: E402  (runpod is pinned)

from worker_config import InputError, JobSpec, output_key, parse_input  # noqa: E402
import multi_image  # noqa: E402
import picture as picture_prep  # noqa: E402
import views  # noqa: E402

MODEL_DIR = "/models/TRELLIS.2-4B"
HDRI_PATH = "/app/assets/hdri/studio.exr"  # CC0 (Poly Haven, see assets/hdri/license.txt)
PRESIGN_TTL_SECONDS = 7 * 24 * 3600
CACHE_CONTROL = "public, max-age=31536000"

# Turntable: one full turn at a fixed, slightly raised camera (the official render_video bobs the
# camera up and down, which reads as a wobble in a looping grid tile). It starts from the view that
# best matches the input picture (views.py) — frame 0 is the poster every grid shows — and the app's
# 3D viewer opens at the same angle (the model's `view`).
TURNTABLE_SIZE = 768
TURNTABLE_FRAMES = 240  # one turn in 8 s — slow enough to read the object in a grid tile
TURNTABLE_FPS = 30
TURNTABLE_PITCH = 0.30  # radians above the horizon
TURNTABLE_FOV = 40
# The camera is fitted to each object: its bounding sphere fills the frame with this much room to spare,
# so it never leaves the frame as it turns and small objects are not lost in the middle.
FRAME_MARGIN = 1.06
VIEW_SEARCH_SIZE = 192

# Time budget. The endpoint stops a job at 900 s (OpenSourceGen sets the same per run); the handler gives
# up a little earlier and asks RunPod for a fresh worker, because a RunPod timeout fails the job but leaves
# a stuck handler running — the worker then never takes another job.
JOB_DEADLINE_S = 840
# The GLB bake runs in its own process (bake.py) so it can be stopped: it usually takes 40–110 s, but on
# some shapes (fragmented ones: hair, fur) it runs for tens of minutes — for one picture even the cheaper
# bake without the remesh stalled. So three tries: the official bake, the same without the remesh, and a
# vertex-colour GLB (no UV unwrap, no texture bake), which always finishes in seconds.
BAKE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bake.py")
BAKE_TRIES = (  # (mode, faces, texture, limit s); None = as asked
    ("remesh", None, None, 270),
    ("plain", 250_000, 2048, 120),
    ("vertex", 250_000, 0, 120),
)
# Background: one flat dark grey, #17171a — the app paints the same colour around the model (client
# index.css --m3d-bg). A gradient bands into visible rings once the clip is H.264-compressed.
BG = (23 / 255, 23 / 255, 26 / 255)

_pipeline = None
_envmap = None
_s3 = None


def log(msg: str) -> None:
    print(f"trellis2-4b - {msg}", flush=True)


# --- R2 ---------------------------------------------------------------------------------------

def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def s3_client():
    global _s3
    if _s3 is None:
        import boto3
        from botocore.config import Config

        missing = [n for n in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY", "R2_BUCKET_NAME") if not _env(n)]
        if missing:
            raise RuntimeError(f"R2 is not configured on this endpoint (missing {', '.join(missing)})")
        _s3 = boto3.client(
            "s3",
            endpoint_url=_env("BUCKET_ENDPOINT_URL"),
            aws_access_key_id=_env("BUCKET_ACCESS_KEY_ID"),
            aws_secret_access_key=_env("BUCKET_SECRET_ACCESS_KEY"),
            region_name="auto",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, retries={"max_attempts": 5, "mode": "standard"}),
        )
    return _s3


def upload(path: str, key: str, content_type: str) -> str:
    bucket = _env("R2_BUCKET_NAME")
    s3_client().upload_file(path, bucket, key, ExtraArgs={"ContentType": content_type, "CacheControl": CACHE_CONTROL})
    return s3_client().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=PRESIGN_TTL_SECONDS)


def download_input(key: str, dest: str) -> None:
    bucket = _env("R2_INPUT_BUCKET_NAME") or _env("R2_BUCKET_NAME")
    try:
        s3_client().download_file(bucket, key, dest)
    except Exception as exc:  # botocore ClientError (404/403) or transport errors
        raise InputError(f"could not read the input picture ({type(exc).__name__})") from exc


# --- Model ------------------------------------------------------------------------------------

def load():
    """Load the pipeline and the studio light once per worker (cold start)."""
    global _pipeline, _envmap
    if _pipeline is not None:
        return _pipeline, _envmap
    import cv2
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap

    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_DIR)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Upstream's default: each stage's model goes to the GPU only for that stage (about a second of
    # PCIe per job). Keeping all of them resident left too little for the GLB bake of a High detail
    # model on a 48 GB card (CuMesh out of memory).
    pipeline.low_vram = True
    pipeline.cuda()
    hdri = cv2.cvtColor(cv2.imread(HDRI_PATH, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    _envmap = EnvMap(torch.tensor(hdri, dtype=torch.float32, device="cuda"))
    _pipeline = pipeline
    log(f"pipeline ready in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)} ({vram_gb:.0f} GB, low_vram={pipeline.low_vram})")
    return _pipeline, _envmap


def open_picture(path: str):
    from PIL import Image, ImageOps

    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img.load()
    except Exception as exc:
        raise InputError("the input is not a readable picture") from exc
    if img.width < 64 or img.height < 64:
        raise InputError("the picture is too small (at least 64 x 64)")
    if not (img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info):
        return img.convert("RGB")
    # Transparency: drop an invisible residue of the old background, or ignore an alpha that cuts nothing
    # out (picture.py) — TRELLIS.2 uses any alpha as the cut-out as it is.
    import numpy as np

    cleaned = picture_prep.clean_alpha(np.asarray(img.convert("RGBA")))
    if cleaned is None:
        log("alpha cuts nothing out — removing the background instead")
        return img.convert("RGB")
    return Image.fromarray(cleaned, "RGBA")


def cut_out(pipeline, img):
    """The pipeline's own preprocessing (background removal, crop to the object, 1024 max)."""
    try:
        return pipeline.preprocess_image(img)
    except ValueError as exc:  # np.min of an empty array: no foreground at all
        raise InputError("no object was found in the picture — use a picture of one clear subject") from exc


def orbit_cameras(mesh, yaws, pitch: float):
    """Cameras around the object's own centre, at the distance where its bounding sphere fits the frame."""
    import math
    import torch
    import utils3d

    v = mesh.vertices.float()
    center = (v.min(0).values + v.max(0).values) / 2
    radius = float((v - center).norm(dim=1).max())
    fov = math.radians(TURNTABLE_FOV)
    dist = radius / math.sin(fov / 2) * FRAME_MARGIN
    up = torch.tensor([0.0, 0.0, 1.0], device=v.device)
    fov_t = torch.tensor(fov, device=v.device)
    intr = utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t)
    extr = []
    for yaw in yaws:
        d = torch.tensor([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)], device=v.device)
        extr.append(utils3d.torch.extrinsics_look_at(center + d * dist, center, up))
    # Near / far around the object (the renderer's defaults, 1..100, would clip a close camera).
    options = {"near": max(0.01, dist - radius * 1.5), "far": dist + radius * 1.5}
    return extr, [intr] * len(yaws), options


def choose_views(mesh, envmap, pictures: list) -> list[tuple[float, float]]:
    """For each cut-out input, the yaw whose small render best matches it (views.py), as (yaw, score).
    The model is rendered once; every picture is scored against the same renders."""
    import numpy as np
    from trellis2.utils import render_utils

    targets = [views.target_from_cutout(np.asarray(p.convert("RGB")).astype(np.float32) / 255.0) for p in pictures]
    yaws = views.candidate_yaws()
    if all(t is None for t in targets):
        return [(yaws[0], -1.0)] * len(pictures)
    extr, intr, opts = orbit_cameras(mesh, yaws, TURNTABLE_PITCH)
    frames = render_utils.render_frames(mesh, extr, intr, {"resolution": VIEW_SEARCH_SIZE, "ssaa": 1, "peel_layers": 2, **opts}, verbose=False, envmap=envmap)
    rendered = [views.normalize_view(shaded.astype(np.float32) / 255.0, mask[..., 0] > 127) for shaded, mask in zip(frames["shaded"], frames["mask"])]
    return [(yaws[0], -1.0) if t is None else views.best_yaw(t, rendered, yaws) for t in targets]


def render_turntable(mesh, envmap, out_path: str, yaw0: float) -> dict:
    import imageio
    import numpy as np
    import torch
    from trellis2.utils import render_utils

    yaws = (yaw0 - torch.linspace(0, 2 * np.pi, TURNTABLE_FRAMES + 1)[:-1]).tolist()
    extr, intr, opts = orbit_cameras(mesh, yaws, TURNTABLE_PITCH)
    frames = render_utils.render_frames(mesh, extr, intr, {"resolution": TURNTABLE_SIZE, **opts}, verbose=False, envmap=envmap)

    # The renderer leaves the background black (shading is accumulated over coverage), so the background
    # goes in by the anti-aliased coverage mask of the nearest surface.
    n = TURNTABLE_SIZE
    bg = np.array(BG, np.float32)

    writer = imageio.get_writer(out_path, fps=TURNTABLE_FPS, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                                macro_block_size=16)
    try:
        for shaded, mask in zip(frames["shaded"], frames["mask"]):
            a = mask[..., :1].astype(np.float32) / 255.0
            rgb = shaded.astype(np.float32) / 255.0 + bg * (1 - a)
            writer.append_data((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    finally:
        writer.close()
    return {"width": n, "height": n, "fps": TURNTABLE_FPS, "frames": TURNTABLE_FRAMES}


def free_gpu_cache() -> None:
    """Hand PyTorch's cached blocks back to the driver: CuMesh / nvdiffrast allocate with cudaMalloc
    directly and cannot use PyTorch's reserve."""
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()


class BakeTooSlow(RuntimeError):
    pass


def _last_line(text) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    lines = [line.strip() for line in (text or "").replace("\r", "\n").splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


def bake_glb(pipeline, mesh, spec: JobSpec, out_path: str, deadline: float, tries_log: list) -> dict:
    """The official bake (o_voxel.postprocess.to_glb, as app.py) in bake.py, under a time limit, then the
    cheaper tries. Returns {faces, vertices, texture_size, bake}. `tries_log` gets one entry per try
    (mode, result, seconds, the last step it printed) so a stall can be told apart from a slow bake."""
    import torch

    free_gpu_cache()
    saved = out_path + ".mesh.pt"
    torch.save({
        "vertices": mesh.vertices.detach().cpu(), "faces": mesh.faces.detach().cpu(),
        "attrs": mesh.attrs.detach().cpu(), "coords": mesh.coords.detach().cpu(),
        "layout": pipeline.pbr_attr_layout, "voxel_size": mesh.voxel_size,
    }, saved)
    for mode, faces, texture, limit_s in BAKE_TRIES:
        faces = faces or spec.decimation_target
        texture = spec.texture_size if texture is None else texture
        limit = min(limit_s, deadline - time.time() - 20)
        if limit < 30:
            break
        t = time.time()
        try:
            p = subprocess.run([sys.executable, "-u", BAKE_SCRIPT, saved, out_path, str(faces), str(texture), mode],
                               capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired as exc:
            last = _last_line(exc.stdout) or _last_line(exc.stderr)
            tries_log.append({"mode": mode, "result": "timeout", "s": round(time.time() - t, 1), "last": last})
            log(f"bake {mode} ({faces} faces, {texture}px) still running after {limit:.0f}s — stopped at: {last}")
            continue
        if p.returncode != 0:
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-12:]
            tries_log.append({"mode": mode, "result": "error", "s": round(time.time() - t, 1), "last": tail[-1][:200] if tail else ""})
            log(f"bake {mode} failed:\n" + "\n".join(tail))
            if any("out of memory" in line.lower() for line in tail):
                continue  # a cheaper try may fit
            continue
        counts = json.loads(p.stdout.strip().splitlines()[-1])
        tries_log.append({"mode": mode, "result": "ok", "s": round(time.time() - t, 1)})
        return {**counts, "texture_size": texture, "bake": mode}
    err = BakeTooSlow("the GLB bake ran out of time")
    err.tries = tries_log
    raise err


# --- Job --------------------------------------------------------------------------------------

def run_job(job: dict) -> dict:
    import torch

    started = time.time()
    deadline = started + JOB_DEADLINE_S
    spec = parse_input(job.get("input"))
    job_id = str(job.get("id") or uuid.uuid4())
    timings: dict[str, float] = {}

    # Progress goes out on its own thread as an IN_PROGRESS update. One that lands after the job's result
    # puts a finished job back to IN_PROGRESS for good (seen on 4 of ~45 dev jobs: every stage done, files
    # uploaded, status never COMPLETED), so each one is waited for before the result is returned, and the
    # last stage is not reported at all (the result follows it at once).
    updates: list[threading.Thread] = []

    def stage(name: str, started: float, report: bool = True) -> None:
        timings[name] = round(time.time() - started, 2)
        if report:
            update = threading.Thread(target=send_progress, args=(job, {"stage": name, "timings": dict(timings)}), daemon=True)
            update.start()
            updates.append(update)

    pipeline, envmap = load()
    work = tempfile.mkdtemp(prefix="trellis2-")
    try:
        t = time.time()
        pictures = []
        for i, key in enumerate(spec.image_r2_keys):
            src = os.path.join(work, f"input{i}")
            download_input(key, src)
            try:
                pictures.append(cut_out(pipeline, open_picture(src)))
            except InputError as exc:
                if len(spec.image_r2_keys) == 1:
                    raise
                raise InputError(f"Picture {i + 1}: {exc}") from exc
        stage("prepare", t)

        t = time.time()
        if len(pictures) == 1:
            mesh = pipeline.run(pictures[0], seed=spec.seed, preprocess_image=False, pipeline_type=spec.pipeline_type)[0]
        else:
            generate_timings: dict = {}
            mesh = multi_image.run(pipeline, pictures, spec.seed, spec.pipeline_type, spec.together_steps, steps=spec.sampler_steps, timings=generate_timings)[0]
            log(f"several pictures ({len(pictures)}, steps={spec.sampler_steps}, together_steps={spec.together_steps}): {generate_timings}")
        mesh.simplify(16777216)  # nvdiffrast's limit (as upstream)
        free_gpu_cache()
        stage("generate", t)

        tag = uuid.uuid4().hex[:8]
        result: dict = {"seed": spec.seed, "quality": spec.quality, "timings": timings, "gpu": torch.cuda.get_device_name(0), "pictures": len(pictures)}
        if len(pictures) > 1:
            result["generate_timings"] = generate_timings

        t = time.time()
        # How the app's 3D viewer opens it (model-viewer camera-orbit, degrees) — the turntable's first frame:
        # the main picture's view. The other pictures' views say how well the model took each of them.
        chosen = choose_views(mesh, envmap, pictures)
        input_views = []
        for yaw, match in chosen:
            theta, phi = views.viewer_orbit(yaw, TURNTABLE_PITCH)
            input_views.append({"theta": round(theta, 1), "phi": round(phi, 1), "match": round(match, 3)})
        yaw0, view = chosen[0][0], input_views[0]
        if len(pictures) > 1:
            result["together_steps"] = spec.together_steps
            result["sampler_steps"] = spec.sampler_steps
            result["input_views"] = input_views
        stage("view", t)

        if spec.turntable:
            t = time.time()
            mp4 = os.path.join(work, "turntable.mp4")
            meta = render_turntable(mesh, envmap, mp4, yaw0)
            key = output_key(f"{tag}.mp4", job_id, spec.uid)
            url = upload(mp4, key, "video/mp4")
            result["videos"] = [{"type": "s3_url", "data": url, "r2_key": key, "filename": f"{tag}.mp4", **meta}]
            stage("turntable", t)

        t = time.time()
        glb_path = os.path.join(work, "model.glb")
        bake_tries: list = []
        result["bake_tries"] = bake_tries
        counts = bake_glb(pipeline, mesh, spec, glb_path, deadline, bake_tries)
        stage("bake", t)

        t = time.time()
        key = output_key(f"{tag}.glb", job_id, spec.uid)
        url = upload(glb_path, key, "model/gltf-binary")
        result["models"] = [{
            "type": "s3_url", "data": url, "r2_key": key, "filename": f"{tag}.glb", "format": "glb",
            "bytes": os.path.getsize(glb_path), "view": view, **counts,
        }]
        stage("upload", t, report=False)
        log(f"job {job_id}: {spec.quality} pictures={len(pictures)} seed={spec.seed} faces={counts['faces']} glb={result['models'][0]['bytes'] / 1e6:.1f} MB timings={timings}")
        return result
    finally:
        for update in updates:
            update.join(timeout=15)
        shutil.rmtree(work, ignore_errors=True)
        torch.cuda.empty_cache()


TOO_SLOW = "building this 3D model took too long — try again, or use Standard quality"


def handler(job: dict) -> dict:
    # The job runs on a thread so a hang anywhere (a CUDA call cannot be interrupted) still answers before
    # RunPod's timeout. refresh_worker: RunPod replaces this worker afterwards, which ends the stuck thread;
    # it is also asked for after any unexpected error, since a CUDA error can leave the context unusable.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(run_job, job)
    try:
        return future.result(timeout=JOB_DEADLINE_S + 30)
    except FutureTimeout:
        log(f"job {job.get('id')}: no result after {JOB_DEADLINE_S + 30}s — answering and recycling the worker")
        return {"error": TOO_SLOW, "refresh_worker": True}
    except InputError as exc:
        log(f"job {job.get('id')}: refused: {exc}")
        return {"error": str(exc)}
    except BakeTooSlow as exc:
        log(f"job {job.get('id')}: every bake try ran out of time: {getattr(exc, 'tries', [])}")
        return {"error": TOO_SLOW, "bake_tries": getattr(exc, "tries", []), "refresh_worker": True}
    except Exception as exc:
        import torch

        traceback.print_exc()
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return {"error": "the GPU ran out of memory on this picture — try Standard quality", "refresh_worker": True}
        return {"error": f"3D generation failed: {type(exc).__name__}: {exc}"[:500], "refresh_worker": True}
    finally:
        pool.shutdown(wait=False)


if __name__ == "__main__":
    load()  # cold start before the first job, not inside it
    runpod.serverless.start({"handler": handler})
