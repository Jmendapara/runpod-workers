"""RunPod serverless handler: one picture → a textured, PBR 3D model with TRELLIS.2-4B.

Per job:
  1. download the picture from R2 (input contract: worker_config.py)
  2. TRELLIS.2 image-to-3D, the official pipeline and sampler settings (pipeline.json); the
     background is removed with BiRefNet unless the picture already has transparency
  3. bake the GLB exactly as the official demo does (o_voxel.postprocess.to_glb: remesh, UV
     unwrap, base colour + metallic/roughness textures, WebP)
  4. render a 360° turntable of the model under a studio light (the app's grid preview)
  5. upload both to R2 and return them as { type: 's3_url' } items

Output:
  { "models": [{ type, data, r2_key, filename, format: 'glb', bytes, faces, vertices, texture_size }],
    "videos": [{ type, data, r2_key, filename, width, height, fps, frames }],
    "seed": int, "timings": { stage: seconds }, "gpu": str }

R2 is required (a GLB is tens of MB): BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID,
BUCKET_SECRET_ACCESS_KEY, R2_BUCKET_NAME (+ optional R2_INPUT_BUCKET_NAME).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod  # noqa: E402

from worker_config import InputError, JobSpec, output_key, parse_input  # noqa: E402
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
TURNTABLE_FRAMES = 120
TURNTABLE_FPS = 30
TURNTABLE_PITCH = 0.30  # radians above the horizon
TURNTABLE_FOV = 40
# The camera is fitted to each object: its bounding sphere fills the frame with this much room to spare,
# so it never leaves the frame as it turns and small objects are not lost in the middle.
FRAME_MARGIN = 1.06
VIEW_SEARCH_SIZE = 192
# Background: a soft dark-grey vignette that sits well in both site themes.
BG_CENTER = (0.165, 0.165, 0.180)
BG_EDGE = (0.075, 0.075, 0.085)

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
    return img.convert("RGBA") if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info else img.convert("RGB")


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


def choose_view(mesh, envmap, picture) -> tuple[float, float]:
    """The yaw whose small render best matches the cut-out input (views.py). Returns (yaw, score)."""
    import numpy as np
    from trellis2.utils import render_utils

    target = views.target_from_cutout(np.asarray(picture.convert("RGB")).astype(np.float32) / 255.0)
    yaws = views.candidate_yaws()
    if target is None:
        return yaws[0], -1.0
    extr, intr, opts = orbit_cameras(mesh, yaws, TURNTABLE_PITCH)
    frames = render_utils.render_frames(mesh, extr, intr, {"resolution": VIEW_SEARCH_SIZE, "ssaa": 1, "peel_layers": 2, **opts}, verbose=False, envmap=envmap)
    rendered = [views.normalize_view(shaded.astype(np.float32) / 255.0, mask[..., 0] > 127) for shaded, mask in zip(frames["shaded"], frames["mask"])]
    return views.best_yaw(target, rendered, yaws)


def render_turntable(mesh, envmap, out_path: str, yaw0: float) -> dict:
    import imageio
    import numpy as np
    import torch
    from trellis2.utils import render_utils

    yaws = (yaw0 - torch.linspace(0, 2 * np.pi, TURNTABLE_FRAMES + 1)[:-1]).tolist()
    extr, intr, opts = orbit_cameras(mesh, yaws, TURNTABLE_PITCH)
    frames = render_utils.render_frames(mesh, extr, intr, {"resolution": TURNTABLE_SIZE, **opts}, verbose=False, envmap=envmap)

    # The renderer leaves the background black (shading is accumulated over coverage), so the vignette goes
    # in by the anti-aliased coverage mask of the nearest surface.
    n = TURNTABLE_SIZE
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.clip(np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n * 0.72), 0, 1)[..., None]
    bg = np.array(BG_CENTER, np.float32) * (1 - d) + np.array(BG_EDGE, np.float32) * d

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


def bake_glb(pipeline, mesh, spec: JobSpec, out_path: str) -> dict:
    import o_voxel

    free_gpu_cache()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=spec.decimation_target,
        texture_size=spec.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(out_path, extension_webp=True)
    return {"faces": int(len(glb.faces)), "vertices": int(len(glb.vertices))}


# --- Job --------------------------------------------------------------------------------------

def run_job(job: dict) -> dict:
    import torch

    spec = parse_input(job.get("input"))
    job_id = str(job.get("id") or uuid.uuid4())
    timings: dict[str, float] = {}

    def stage(name: str, started: float) -> None:
        timings[name] = round(time.time() - started, 2)
        runpod.serverless.progress_update(job, {"stage": name, "timings": timings})

    pipeline, envmap = load()
    work = tempfile.mkdtemp(prefix="trellis2-")
    try:
        t = time.time()
        src = os.path.join(work, "input")
        download_input(spec.image_r2_key, src)
        picture = cut_out(pipeline, open_picture(src))
        stage("prepare", t)

        t = time.time()
        mesh = pipeline.run(picture, seed=spec.seed, preprocess_image=False, pipeline_type=spec.pipeline_type)[0]
        mesh.simplify(16777216)  # nvdiffrast's limit (as upstream)
        free_gpu_cache()
        stage("generate", t)

        tag = uuid.uuid4().hex[:8]
        result: dict = {"seed": spec.seed, "quality": spec.quality, "timings": timings, "gpu": torch.cuda.get_device_name(0)}

        t = time.time()
        yaw0, match = choose_view(mesh, envmap, picture)
        theta, phi = views.viewer_orbit(yaw0, TURNTABLE_PITCH)
        # How the app's 3D viewer opens it (model-viewer camera-orbit, degrees) — the turntable's first frame.
        view = {"theta": round(theta, 1), "phi": round(phi, 1), "match": round(match, 3)}
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
        counts = bake_glb(pipeline, mesh, spec, glb_path)
        stage("bake", t)

        t = time.time()
        key = output_key(f"{tag}.glb", job_id, spec.uid)
        url = upload(glb_path, key, "model/gltf-binary")
        result["models"] = [{
            "type": "s3_url", "data": url, "r2_key": key, "filename": f"{tag}.glb", "format": "glb",
            "bytes": os.path.getsize(glb_path), "texture_size": spec.texture_size, "view": view, **counts,
        }]
        stage("upload", t)
        log(f"job {job_id}: {spec.quality} seed={spec.seed} faces={counts['faces']} glb={result['models'][0]['bytes'] / 1e6:.1f} MB timings={timings}")
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)
        torch.cuda.empty_cache()


def handler(job: dict) -> dict:
    try:
        return run_job(job)
    except InputError as exc:
        log(f"job {job.get('id')}: refused: {exc}")
        return {"error": str(exc)}
    except Exception as exc:
        import torch

        traceback.print_exc()
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return {"error": "the GPU ran out of memory on this picture — try Standard quality"}
        return {"error": f"3D generation failed: {type(exc).__name__}: {exc}"[:500]}


if __name__ == "__main__":
    load()  # cold start before the first job, not inside it
    runpod.serverless.start({"handler": handler})
