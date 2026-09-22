# runpod-workers

Consolidated RunPod serverless ComfyUI workers. One shared base image (CUDA
13.0.3 + PyTorch cu130 + ComfyUI + the universal handler) plus a small
declarative `model.yaml` per model.

## Adding a new model

```bash
mkdir models/my-new-thing
$EDITOR models/my-new-thing/model.yaml
```

There is no per-model Dockerfile — `build.sh` generates one automatically from
`model.yaml` at build time (one image layer per downloaded weight; see
[Building images](#building-images)).

Minimum viable `model.yaml`:

```yaml
name: my-new-thing
output:
  type: image          # or video / audio / image+audio / image+video / image+video+gifs
```

Add `custom_nodes:`, `post_install:`, `extra_model_paths_additions:`, `env:`
as needed. See `schema/model.schema.json` for the full contract.

## Building images

Builds happen on a Hetzner box via `curl | bash`. The script self-clones the
repo — no local checkout needed on the build host. **No CI does Docker
builds.** Only schema validation runs in GitHub Actions on PRs.

```bash
export DOCKERHUB_USERNAME=jmendapara
export DOCKERHUB_TOKEN=...
curl -fsSL https://raw.githubusercontent.com/Jmendapara/runpod-workers/main/build.sh | MODEL=wan-animate bash
```

**Important:** env vars before `curl` only apply to `curl`, not the piped `bash`. Put `MODEL=...` directly before `bash`, or `export MODEL=...` first. `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` must be `export`ed so they cross the pipe.

`MODEL` values:

- `base` — build & push the shared base image
- `<model-name>` — build & push one model image
- `all` — build base, then every model

Tag format pushed: `jmendapara/<model>-runpod-worker:YYYY-MM-DD-HHMM-<shortsha>`
(immutable; never `:latest`).

Model builds auto-discover the most recent base image tag from Docker Hub.
Override with `BASE_TAG=YYYY-MM-DD-HHMM-<sha>`.

### Sharded builds (automatic, all models)

`build.sh` generates each model's Dockerfile on the fly from its `model.yaml`,
baking **every `model_downloads` entry in its own `RUN`/layer**. No single
layer holds more than one weight file, so each pushed blob stays inside Docker
Hub's upload-session / BuildKit lease window no matter how large the model is.
This is what prevents `blob upload invalid - upload state expired` /
`lease does not exist` on big images — and it applies to every model, current
and future, with nothing to maintain per model. Shard count = number of
downloads (capped at 50 layers; beyond that they round-robin).

## Building remotely from your laptop

`tools/remote-build.sh` does the Hetzner dance for you: it logs into the build
box with the credentials in `.env`, ships this checkout's `build.sh` over,
launches it **detached** (it survives Ctrl-C, laptop sleep and dropped Wi-Fi)
and streams the log live. You never open the Hetzner console or an SSH
session by hand.

One-time setup:

```bash
cp .env.example .env && chmod 600 .env
$EDITOR .env        # HETZNER_HOST, HETZNER_PASSWORD, DOCKERHUB_TOKEN, HF/Civitai tokens
```

`.env` is gitignored — never commit it. Then:

```bash
./tools/remote-build.sh                   # pick a target from a menu
./tools/remote-build.sh wan-animate       # build + push one model
./tools/remote-build.sh base              # rebuild the shared base image
./tools/remote-build.sh all               # base, then every model
```

While a build runs:

- **Ctrl-C only stops watching**; the build keeps going on the box.
- `./tools/remote-build.sh logs` reattaches to the live log (auto-reconnects on drops).
- `./tools/remote-build.sh status` shows what is running, for how long, and disk free.
- `./tools/remote-build.sh stop` cancels the running build (`--force` to SIGKILL).

Useful flags: `--branch <b>` (default: your current local branch, which must
be pushed), `--no-push`, `--base-tag <t>`, `--dry-run` (exercises the whole
pipeline without building anything), `-y` (skip the confirmation),
`--reset-hostkey` (after a Hetzner reinstall/rescue). `--help` lists the rest.

Auth: the password from `HETZNER_PASSWORD` is handed to OpenSSH through its
askpass hook (no `sshpass` needed). Set `HETZNER_SSH_KEY=~/.ssh/id_ed25519`
to use a key instead; `./tools/remote-build.sh setup` installs your public key
on the box so you can switch.

Secrets travel over the SSH channel into a 0600 tmpfs file that the remote
wrapper sources and deletes before the build starts — never in argv, shell
history or on disk. The box logs out of Docker Hub when the build ends.

## Updating ComfyUI

```bash
curl -fsSL .../build.sh | COMFYUI_VERSION=latest MODEL=base bash   # rebuild base (CUDA 13.0.3 / torch cu130)
curl -fsSL .../build.sh | MODEL=all bash                           # rebuild all models on new base
```

`COMFYUI_VERSION` takes `latest` (newest stable release) or `nightly` (master
HEAD) — comfy-cli does not accept a commit SHA.

The base's CUDA/torch stack defaults live in `base/Dockerfile` (currently
`nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04` + `--cuda-version 13.0` +
`https://download.pytorch.org/whl/cu130`). CUDA >= 13 is required for ComfyUI's
comfy-kitchen CUDA kernels (int8_convrot / nvfp4 weights, e.g. minimax-h3); on
cu128 those run on a slow dequant fallback. To roll a base back to cu128 without
editing the Dockerfile:

```bash
curl -fsSL .../build.sh | CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
  CUDA_VERSION_FOR_COMFY=12.8 PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  MODEL=base bash
```

### Base image compatibility

| Model | Base | Notes |
|---|---|---|
| `minimax-h3` | cu130 base built from a Dockerfile that installs `gcc` + `libc6-dev` (2026-09-22 or later) | Needs ComfyUI >= 0.30.0. Triton JIT-compiles kernel launchers at runtime; without a C compiler the MiniMax node fails with "Failed to find C compiler" |
| `scail-2` | **pin `BASE_TAG=2026-07-07-1526-510c57c`** until validated on cu130 | `extra_pip: cupy-cuda12x` → needs `cupy-cuda13x` on a cu130 base |
| `wan-animate` | **pin `BASE_TAG=2026-07-07-1526-510c57c`** until validated on cu130 | `onnxruntime-gpu` is pinned for CUDA 12.8 (commit 17d0400); needs a CUDA 13 build |
| `ltx-2.3` | **pin `BASE_TAG=2026-07-07-1526-510c57c`** until validated on cu130 | `pip_extras: sageattention` untested against torch cu130 |
| everything else | untested on cu130 — pin the old tag until rebuilt + smoke-tested | |

Model builds auto-discover the *newest* base tag, so any model rebuilt without
`BASE_TAG=` lands on the cu130 base. Already-deployed endpoint images are not
affected until you rebuild them.

## Validating a `model.yaml` locally

```bash
pip install PyYAML jsonschema
python tools/validate_yaml.py models/*/model.yaml
```

## Smoke testing against a RunPod endpoint

```bash
export RUNPOD_API_KEY=...
python tests/run_smoke.py wan-animate <endpoint-id>
python tests/run_smoke.py minimax-h3 <endpoint-id>   # ~5 s 768x768 clip with audio
```

Smoke inputs live at `tests/smoke/<model>.json` — replace the placeholder
workflow with a known-good ComfyUI workflow JSON for each model before use.

## Rolling a new image onto a RunPod endpoint

Endpoint templates named `…__template__…` are invisible to the REST `/v1/templates`
API, and swapping the template alone does NOT replace idle/FlashBoot workers — jobs keep
landing on the old image until every worker is recycled.

```bash
export RUNPOD_API_KEY=...
# 1. swap the template image (dry run without -y; --expect-name guards against the wrong endpoint id)
tools/runpod-set-template-image.py <endpoint-id> jmendapara/<model>-runpod-worker:<tag> --expect-name "PD - Z Image Turbo - Dev" -y
# 2. drain + restore workers so every worker cold-starts on the new image
tools/runpod-recycle-workers.sh <endpoint-id>
# 3. smoke it
python tests/run_smoke.py <model> <endpoint-id>
```

## Layout

```
base/                       Shared image: handler.py, lib/, runtime/, scripts/, Dockerfile
models/<name>/
  model.yaml                The declarative config that drives the build + handler
                            (build.sh generates the Dockerfile from it — none on disk)
  patches/                  Optional per-model build-time patches
schema/model.schema.json    JSON Schema for model.yaml (single source of truth)
tools/validate_yaml.py      Lint runner (also the CI check)
tools/remote-build.sh       Build on the Hetzner box from your laptop (see above)
tools/runpod-set-template-image.py  Swap an endpoint template's image (GraphQL saveTemplate)
tools/runpod-recycle-workers.sh     Drain + restore an endpoint's workers after a swap
.env.example                Template for the gitignored .env used by remote-build.sh
tests/                      Smoke tests
build.sh                    The one entrypoint, curl|bash-friendly
```

## minimax-h3 endpoint sizing

Image ≈ 12.6 GB base + 44.5 GB weights. Suggested RunPod serverless settings:

- GPU: H100 80GB (alt: RTX PRO 6000 Blackwell 96GB). ~45 GB of weights stay
  resident, so 48 GB cards are too tight for 15 s / 1344x768 clips.
- Container disk: 80 GB. No network volume (weights are baked in).
- Workers: min 0 / max 1 to start. FlashBoot on. Execution timeout >= 1200 s
  (15 s clips at 8 steps).
- No start command — the image runs its built-in `/start.sh`.
- Env: only the four R2 vars below (`BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`,
  `BUCKET_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`).
- License: MiniMax H3 weights are under the minimax-h3-community-license-agreement;
  Comfy states commercial use of locally generated outputs needs a MiniMax
  commercial license (sold via Comfy).

## Runtime env vars (set on the RunPod endpoint)

R2 upload (optional — unset for base64 responses):
- `BUCKET_ENDPOINT_URL`
- `BUCKET_ACCESS_KEY_ID`
- `BUCKET_SECRET_ACCESS_KEY`
- `R2_BUCKET_NAME`
- `R2_INPUT_BUCKET_NAME` (optional; defaults to `R2_BUCKET_NAME`)

Worker tuning:
- `REFRESH_WORKER=true` to recycle the worker after each job
- `NETWORK_VOLUME_DEBUG=true` (default) for `/runpod-volume` diagnostics
- `COMFY_LOG_LEVEL=DEBUG` (default), `COMFY_RESTART_DELAY=5`, `COMFY_MAX_RAPID_RESTARTS=5`
- `WEBSOCKET_RECONNECT_ATTEMPTS=5`, `WEBSOCKET_RECONNECT_DELAY_S=3`
- `WEBSOCKET_TRACE=true` to enable websocket-client trace logging
