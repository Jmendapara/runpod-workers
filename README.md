# runpod-workers

Consolidated RunPod serverless ComfyUI workers. One shared base image (CUDA
12.8.1 + PyTorch cu128 + ComfyUI + the universal handler) plus a small
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

## Updating ComfyUI

```bash
curl -fsSL .../build.sh | COMFYUI_VERSION=0.4.5 MODEL=base bash    # rebuild base
curl -fsSL .../build.sh | MODEL=all bash                           # rebuild all models on new base
```

## Validating a `model.yaml` locally

```bash
pip install PyYAML jsonschema
python tools/validate_yaml.py models/*/model.yaml
```

## Smoke testing against a RunPod endpoint

```bash
export RUNPOD_API_KEY=...
python tests/run_smoke.py wan-animate <endpoint-id>
```

Smoke inputs live at `tests/smoke/<model>.json` — replace the placeholder
workflow with a known-good ComfyUI workflow JSON for each model before use.

## Layout

```
base/                       Shared image: handler.py, lib/, runtime/, scripts/, Dockerfile
models/<name>/
  model.yaml                The declarative config that drives the build + handler
                            (build.sh generates the Dockerfile from it — none on disk)
  patches/                  Optional per-model build-time patches
schema/model.schema.json    JSON Schema for model.yaml (single source of truth)
tools/validate_yaml.py      Lint runner (also the CI check)
tests/                      Smoke tests
build.sh                    The one entrypoint, curl|bash-friendly
```

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
