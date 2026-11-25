# trellis2-4b — image to 3D (TRELLIS.2-4B)

[microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT) on RunPod serverless: one picture
in, a textured PBR GLB plus a 360° turntable MP4 out. Not a ComfyUI worker — the official pipeline
runs as-is, with the official sampler settings (`pipeline.json`) and the official GLB bake
(`o_voxel.postprocess.to_glb`, as in `app.py`). Used by OpenSourceGen's **Image to 3D** tool
(`shared/workflows/image-to-3d.js`).

## Build

```bash
./tools/remote-build.sh trellis2-4b      # or: MODEL=trellis2-4b bash build.sh (repo root)
```

Needs `HUGGINGFACE_ACCESS_TOKEN` with access to the gated
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
(TRELLIS.2's image encoder; accept its licence on the token's account). Every upstream repo and Hugging
Face revision is pinned in the Dockerfile; the CUDA extensions (nvdiffrast, nvdiffrec, CuMesh,
FlexGEMM, o-voxel) are compiled for sm 8.0/8.6/8.9/9.0. Weights (~17 GB) are baked into the image.

Background removal uses `ZhengPeng7/BiRefNet` (MIT, the code's default) instead of `briaai/RMBG-2.0`
(CC BY-NC, gated), which the published `pipeline.json` names — the Dockerfile rewrites that entry.

## Endpoint

- GPU: 48 GB+ (L40S / A6000 Ada / A100 / H100). It runs on 24 GB in the pipeline's low-VRAM mode,
  but High quality can run out of memory there. No Blackwell pools (the flash-attn wheel has no sm_120).
- Container disk 80 GB, FlashBoot on, execution timeout 900 s (the handler answers by 870 s and recycles a stuck worker).
- Env: the four R2 vars only — `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`,
  `BUCKET_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` (+ optional `R2_INPUT_BUCKET_NAME`). R2 is required.

## Input

| Field | Default | |
|---|---|---|
| `image_r2_key` | — | the picture (PNG/JPEG/WebP; transparency is used as the cut-out) |
| `image_r2_keys` | — | instead: 1–4 pictures of the same object from different sides, the main view first (see below) |
| `together_steps` | 0 for two pictures, 4 for more | several pictures: how many last sampler steps use all of them at once |
| `sampler_steps` | 12 (`pipeline.json`) | several pictures: sampler steps of every stage |
| `quality` | `standard` | `standard` = 1024³ cascade (upstream default), `high` = 1536³ cascade |
| `seed` | 42 | |
| `texture_size` | 2048 | 1024 / 2048 / 4096 |
| `decimation_target` | 500000 | faces kept in the GLB (50k–1M) |
| `turntable` | true | also render the 360° MP4 |
| `uid` | — | outputs go under `users/{uid}/generations/` |

## Output

```json
{ "models": [{ "type": "s3_url", "data": "<presigned>", "r2_key": "…/ab12cd34.glb", "format": "glb",
               "bytes": 15000000, "faces": 500000, "vertices": 251000, "texture_size": 2048 }],
  "videos": [{ "type": "s3_url", "data": "<presigned>", "r2_key": "…/ab12cd34.mp4",
               "width": 768, "height": 768, "fps": 30, "frames": 240 }],
  "seed": 42, "quality": "standard", "timings": { "prepare": 1.2, "generate": 17.0, … }, "gpu": "NVIDIA H100 80GB HBM3" }
```

A refused input or a failure returns `{ "error": "…" }`. Unit tests: `python3 tests/test_trellis2.py`.

## Several pictures

TRELLIS.2 has no multi-picture mode; `multi_image.py` ports TRELLIS v1's tuning-free `run_multi_image`
to its samplers (each picture is encoded on its own and the pictures steer every sampler step, in turn or
together). The back, and anything else the main picture does not show, then follows a picture instead of
a guess. Each picture is cut out like a single one; a picture with no object fails the job with
`Picture N: …`. The turntable and `models[0].view` start from the main picture; `input_views` gives the
view each picture was matched to, `generate_timings` the seconds per stage. Measured behaviour, limits
and why the defaults are what they are: the docstring of `multi_image.py`.

## Time budget and fallbacks

The endpoint stops a job at 900 s; the handler answers by 870 s and sets `refresh_worker` (a RunPod
timeout fails the job but leaves a stuck handler running, and the worker never takes another job).
Progress updates are sent on threads the handler waits for before it returns: the SDK's own
`progress_update` is not ordered against the job's result, and one that landed after it left finished
jobs `IN_PROGRESS` for good. The
GLB bake runs in `bake.py` (its own process) under a limit, because on fragmented shapes (hair, fur)
`to_glb` can run for 20+ minutes:

| Try | What | Limit |
|---|---|---|
| `remesh` | the official bake (`to_glb`, remesh, as asked: faces / texture) | 270 s |
| `plain` | `to_glb` without the remesh, 250k faces, 2K texture | 120 s |
| `vertex` | simplified mesh with its colour per vertex — no UV unwrap, no texture | 120 s |

Each try's `{mode, result, s, last}` (`last` = the last step the bake printed) is in the output's
`bake_tries`, also in the error response when every try runs out of time.

## Transparent pictures

TRELLIS.2 uses any alpha channel as the cut-out as it is. `picture.py` first drops alpha under 25 %
(the invisible residue background-removal tools leave, which TRELLIS turns into a flat panel around the
subject), and ignores an alpha that cuts nothing out (under 1 % transparent), so background removal runs. The turntable
(768², 240 frames, one turn in 8 s, flat `#17171a`) starts from the view that best matches the input
(`views.py`); `models[0].view` is that view as a model-viewer camera-orbit.
