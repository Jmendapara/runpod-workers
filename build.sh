#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# runpod-workers builder — single curl|bash entrypoint for Hetzner.
#
# Usage (curl from GitHub, no local checkout required):
#
#   export DOCKERHUB_USERNAME="jmendapara"
#   export DOCKERHUB_TOKEN="..."
#   curl -fsSL https://raw.githubusercontent.com/Jmendapara/runpod-workers/main/build.sh \
#     | MODEL=wan-animate bash
#
# NOTE: `MODEL=...` MUST be on the bash side of the pipe. Putting it before
# `curl` only exports it to curl, not the piped shell. Alternatively
# `export MODEL=...` first, then run the curl pipeline.
#
# Usage (local checkout):
#
#   MODEL=wan-animate bash build.sh
#
# Required env:
#   DOCKERHUB_USERNAME  — your Docker Hub login
#   DOCKERHUB_TOKEN     — your Docker Hub access token
#   MODEL               — "base", "all", or a model name (matches models/<name>/)
#
# Optional env:
#   BRANCH=main                 git branch
#   REPO_URL=...                git remote (default: this repo)
#   IMAGE_NAMESPACE=jmendapara  Docker Hub username/org prefix
#   BASE_TAG=...                Pin a base tag for model builds; else auto-discover
#   NO_PUSH=1                   Build only, skip docker push
#   COMFYUI_VERSION=latest      ComfyUI version (only used for `base` builds)
#   HUGGINGFACE_ACCESS_TOKEN    Passed through to apply_model_config.py for gated repos
#   CIVITAI_TOKEN               Passed through for any model_downloads with auth_header_env=CIVITAI_TOKEN
#
# Tags pushed (NEVER :latest — every build gets a unique immutable tag):
#   base:        jmendapara/runpod-worker-base:<UTC-YYYY-MM-DD-HHMM>-<shortsha>
#   <model>:     jmendapara/<model>-runpod-worker:<UTC-YYYY-MM-DD-HHMM>-<shortsha>
# =============================================================================

REPO_URL="${REPO_URL:-https://github.com/Jmendapara/runpod-workers.git}"
BRANCH="${BRANCH:-main}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-jmendapara}"
COMFYUI_VERSION="${COMFYUI_VERSION:-latest}"
BASE_IMAGE_NAME="${IMAGE_NAMESPACE}/runpod-worker-base"

# BuildKit progress UI is left at BuildKit's default (`auto`): fancy TTY bar
# in interactive shells, plain output in CI. For huge builds (ltx-2.3) where
# the silent "exporting layers" phase is a problem, run with
# `BUILDKIT_PROGRESS=plain` in your environment to see per-layer byte counts.

: "${DOCKERHUB_USERNAME:?Set DOCKERHUB_USERNAME}"
: "${DOCKERHUB_TOKEN:?Set DOCKERHUB_TOKEN}"
: "${MODEL:?Set MODEL=base, MODEL=all, or MODEL=<model-name>}"

echo "============================================="
echo " runpod-workers builder"
echo "============================================="
echo "  Repo:               ${REPO_URL}"
echo "  Branch:             ${BRANCH}"
echo "  Image namespace:    ${IMAGE_NAMESPACE}"
echo "  Target:             MODEL=${MODEL}"
echo "  ComfyUI version:    ${COMFYUI_VERSION} (base builds only)"
echo "  Push:               $([ -z "${NO_PUSH:-}" ] && echo YES || echo NO)"
echo "============================================="

# ---- Step 1: Docker + buildx ----
if ! command -v docker &>/dev/null || ! docker buildx version &>/dev/null 2>&1; then
    echo "[1/5] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
else
    echo "[1/5] Docker already available: $(docker --version)"
fi

if ! docker info &>/dev/null 2>&1; then
    echo "[1/5] Starting Docker daemon..."
    if ! systemctl start docker 2>/dev/null; then
        dockerd &>/dev/null &
        sleep 5
    fi
fi
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not running."; exit 1; }

# ---- Step 2: Docker Hub login ----
echo "[2/5] Logging into Docker Hub as ${DOCKERHUB_USERNAME}..."
echo "${DOCKERHUB_TOKEN}" | docker login --username "${DOCKERHUB_USERNAME}" --password-stdin >/dev/null

# ---- Step 3: Clone monorepo ----
WORK_DIR="/tmp/runpod-workers-build"
[ -d "${WORK_DIR}" ] && rm -rf "${WORK_DIR}"
echo "[3/5] Cloning ${REPO_URL} (branch ${BRANCH}) → ${WORK_DIR}..."
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${WORK_DIR}"
cd "${WORK_DIR}"

SHORT_SHA="$(git rev-parse --short HEAD)"
TS="$(date -u +%Y-%m-%d-%H%M)"
BUILD_TAG="${TS}-${SHORT_SHA}"
echo "       Build tag: ${BUILD_TAG}"

# Free disk before any build
docker system prune -af --volumes 2>/dev/null || true
docker builder prune -af 2>/dev/null || true
echo "       Disk free: $(df -h /var/lib/docker 2>/dev/null | tail -1 | awk '{print $4}' || echo unknown)"

# ---- Helpers ----

# Build & push. Uses buildkit's streaming registry push so the image is never
# materialized as a local tar — important on disk-constrained hosts where the
# image itself (~150 GB for ltx-2.3) already eats most of the available space.
#
# Trade-off: a long push (>60 min) can hit buildkit's internal lease TTL,
# causing "lease does not exist" at the manifest-write step. Mitigations:
#   --provenance=false --sbom=false   skips the attestation layer, removing
#                                     a push round-trip and shaving lease time
#
# If lease expiry recurs and freeing more disk is not an option, the next step
# is enabling Docker's containerd image store (daemon.json
# `features.containerd-snapshotter: true`) so build and push share storage,
# then splitting build/push across two commands. That requires a docker
# daemon restart and is not automated here.
#
# Args:  <tag>  <context-dir>  [extra docker buildx build flags...]
# Honors NO_PUSH: when set, just builds and loads locally (no push).
_build_and_push() {
    local tag="$1"
    local context="$2"
    shift 2
    local extra_args=("$@")

    local push_flag="--push"
    [ -n "${NO_PUSH:-}" ] && push_flag="--load"

    docker buildx build \
        "${extra_args[@]}" \
        --provenance=false --sbom=false \
        ${push_flag} \
        -t "${tag}" \
        "${context}"
}

# Generate a sharded Dockerfile for a model into <out> and echo the shard count.
#
# Each model_download is baked in its OWN RUN step → its own image layer, so no
# single layer ever holds more than one weight file. This keeps every pushed
# blob inside Docker Hub's upload-session / BuildKit lease window regardless of
# the model's total size — which is what prevents the "blob upload invalid -
# upload state expired" / "lease does not exist" failures on large models. This
# applies to EVERY model automatically (current and future); there are no
# per-model Dockerfiles to keep in sync.
#
# Shard count = number of model_downloads, capped at MAX_SHARDS so we never
# approach the OCI ~127-layer limit. Above the cap, downloads round-robin into
# MAX_SHARDS layers (apply_model_config selects index %% N == K per shard);
# below it, the mapping is exactly one file per layer. Models with zero
# downloads get a plain setup→finalize build (no weight layers).
#
# Args: <model_dir> <out_path>   (writes the Dockerfile to <out_path>)
render_model_dockerfile() {
    local model_dir="$1"
    local out="$2"
    local n_dl shards k
    # Count model_downloads without a YAML parser: every download item is a
    # "- source:" line; custom_nodes use "- repo:", so this is exact.
    n_dl=$(grep -cE '^[[:space:]]*-[[:space:]]+source:[[:space:]]' "${model_dir}/model.yaml" 2>/dev/null || true)
    n_dl=${n_dl:-0}

    local MAX_SHARDS=50
    if [ "${n_dl}" -gt "${MAX_SHARDS}" ]; then
        shards=${MAX_SHARDS}
    else
        shards=${n_dl}
    fi

    {
        cat <<'HDR'
# syntax=docker/dockerfile:1.7
# AUTO-GENERATED by build.sh — do not edit by hand. One RUN/layer per weight
# file keeps every pushed blob within Docker Hub's upload window regardless of
# total image size. Customize a model via its model.yaml, never here.
ARG BASE_VERSION
FROM jmendapara/runpod-worker-base:${BASE_VERSION}

COPY model.yaml /etc/worker/model.yaml
COPY . /tmp/model-ctx/

# --- setup: custom nodes + pip (small layer) ---
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase setup
HDR

        if [ "${shards}" -gt 0 ]; then
            printf '\n# --- weight shards: one RUN per download = one independently-pushable layer ---\n'
            k=0
            while [ "${k}" -lt "${shards}" ]; do
                printf 'RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \\\n'
                printf '    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \\\n'
                printf '    CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \\\n'
                printf '    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard %s/%s\n' "${k}" "${shards}"
                k=$((k + 1))
            done
        fi

        cat <<'FIN'

# --- finalize: post_install + env, then drop the build context (small layer) ---
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase finalize \
 && rm -rf /tmp/model-ctx
FIN
    } > "${out}"

    echo "${shards}"
}

# Resolve base tag: explicit env var, or query Docker Hub for the newest
# YYYY-MM-DD-HHMM-<sha> tag on runpod-worker-base.
resolve_base_tag() {
    if [ -n "${BASE_TAG:-}" ]; then
        echo "${BASE_TAG}"
        return 0
    fi
    echo "       Auto-discovering latest base tag from Docker Hub..." >&2
    local tag
    tag=$(curl -fsSL \
        "https://hub.docker.com/v2/repositories/${BASE_IMAGE_NAME}/tags?page_size=100&ordering=last_updated" \
        | python3 -c "
import json, re, sys
data = json.load(sys.stdin)
pat = re.compile(r'^\d{4}-\d{2}-\d{2}-\d{4}-[a-f0-9]+$')
tags = [r['name'] for r in data.get('results', []) if pat.match(r['name'])]
print(sorted(tags, reverse=True)[0] if tags else '', end='')
" || true)
    if [ -z "${tag}" ]; then
        cat <<EOF >&2

ERROR: No base image found at ${BASE_IMAGE_NAME} matching the
       YYYY-MM-DD-HHMM-<sha> tag pattern. Run MODEL=base first to publish a base
       image, or pass BASE_TAG=<tag> to pin an existing one.
EOF
        return 1
    fi
    echo "${tag}"
}

# Build & push base image
build_base() {
    local tag="${BASE_IMAGE_NAME}:${BUILD_TAG}"

    echo "[4/5] Building base image → ${tag}"
    _build_and_push "${tag}" "." \
        --platform linux/amd64 \
        -f base/Dockerfile \
        --build-arg "COMFYUI_VERSION=${COMFYUI_VERSION}"

    echo "============================================="
    echo " ✓ Built$([ -z "${NO_PUSH:-}" ] && echo " and pushed") base image:"
    echo "     ${tag}"
    echo "============================================="
}

# Build & push a model image
build_model() {
    local model="$1"
    local model_dir="models/${model}"

    if [ ! -d "${model_dir}" ]; then
        echo "ERROR: ${model_dir} does not exist. Available models:"
        ls -1 models/ | sed 's/^/  • /'
        return 1
    fi

    echo "[validate] Schema-checking ${model_dir}/model.yaml..."
    if ! python3 tools/validate_yaml.py "${model_dir}/model.yaml" >/dev/null 2>&1; then
        # Re-run with output for the user
        python3 tools/validate_yaml.py "${model_dir}/model.yaml" || true
        echo "ERROR: model.yaml validation failed. Aborting."
        return 1
    fi
    echo "       ✓ model.yaml valid."

    local base_tag
    base_tag="$(resolve_base_tag)" || return 1
    echo "       Using base: ${BASE_IMAGE_NAME}:${base_tag}"

    local tag="${IMAGE_NAMESPACE}/${model}-runpod-worker:${BUILD_TAG}"

    # Forward optional build-time secrets via BuildKit --secret (NOT --build-arg,
    # which would bake the token into image layer history).
    local secret_args=()
    if [ -n "${HUGGINGFACE_ACCESS_TOKEN:-}" ]; then
        secret_args+=(--secret id=hf_token,env=HUGGINGFACE_ACCESS_TOKEN)
    fi
    if [ -n "${CIVITAI_TOKEN:-}" ]; then
        secret_args+=(--secret id=civitai_token,env=CIVITAI_TOKEN)
    fi

    # Generate the sharded Dockerfile from this model's model.yaml (one layer
    # per weight file). Written outside the build context and passed via -f.
    local dockerfile shards
    dockerfile="$(mktemp "/tmp/Dockerfile.${model}.XXXXXX")"
    shards="$(render_model_dockerfile "${model_dir}" "${dockerfile}")"
    echo "       Generated sharded Dockerfile: ${shards} weight layer(s) → ${dockerfile}"

    echo "[4/5] Building model → ${tag}"
    export DOCKER_BUILDKIT=1
    _build_and_push "${tag}" "${model_dir}" \
        --platform linux/amd64 \
        -f "${dockerfile}" \
        --build-arg "BASE_VERSION=${base_tag}" \
        "${secret_args[@]}"

    cat <<EOF
=============================================
 ✓ Built$([ -z "${NO_PUSH:-}" ] && echo " and pushed"):
     ${tag}
   (FROM ${BASE_IMAGE_NAME}:${base_tag})

 → Update the ${model} RunPod endpoint to use image:
     ${tag}
=============================================
EOF
}

# ---- Step 4: Resolve target and build ----

case "${MODEL}" in
    base)
        build_base
        ;;
    all)
        build_base
        for d in models/*/ ; do
            m="$(basename "$d")"
            build_model "${m}"
        done
        ;;
    *)
        build_model "${MODEL}"
        ;;
esac

echo
echo "[5/5] Done."
