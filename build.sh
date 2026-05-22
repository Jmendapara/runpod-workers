#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# runpod-workers builder — single curl|bash entrypoint for Hetzner.
#
# Usage (curl from GitHub, no local checkout required):
#
#   export DOCKERHUB_USERNAME="jmendapara"
#   export DOCKERHUB_TOKEN="..."
#   export MODEL="wan-animate"       # or "base" or "all"
#   curl -fsSL https://raw.githubusercontent.com/Jmendapara/runpod-workers/main/build.sh | bash
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
    local push_flag="--push"
    [ -n "${NO_PUSH:-}" ] && push_flag="--load"

    echo "[4/5] Building base image → ${tag}"
    docker buildx build \
        --platform linux/amd64 \
        -f base/Dockerfile \
        --build-arg "COMFYUI_VERSION=${COMFYUI_VERSION}" \
        ${push_flag} \
        -t "${tag}" \
        .

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
    local push_flag="--push"
    [ -n "${NO_PUSH:-}" ] && push_flag="--load"

    echo "[4/5] Building model → ${tag}"
    docker buildx build \
        --platform linux/amd64 \
        --build-arg "BASE_VERSION=${base_tag}" \
        ${push_flag} \
        -t "${tag}" \
        "${model_dir}"

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
