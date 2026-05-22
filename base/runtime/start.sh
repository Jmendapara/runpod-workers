#!/usr/bin/env bash
set -uo pipefail

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
[ -n "${TCMALLOC}" ] && export LD_PRELOAD="${TCMALLOC}"

# Source any per-model env exports written by apply_model_config.py
if [ -f /etc/worker/env ]; then
    set -a
    # shellcheck disable=SC1091
    . /etc/worker/env
    set +a
fi

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

# ---------- Diagnostics: network-volume detection ----------
echo "worker-comfyui: Detecting network volume..."
echo "  /runpod-volume exists: $([ -d /runpod-volume ] && echo YES || echo NO)"
echo "  /runpod-volume/models exists: $([ -d /runpod-volume/models ] && echo YES || echo NO)"
echo "  /workspace exists: $([ -d /workspace ] && echo YES || echo NO)"
echo "  /workspace/models exists: $([ -d /workspace/models ] && echo YES || echo NO)"
ls -la /runpod-volume/models/ 2>/dev/null || echo "  (cannot list /runpod-volume/models/)"
ls -la /workspace/models/ 2>/dev/null || echo "  (cannot list /workspace/models/)"

# ---------- Pre-launch diagnostics ----------
echo "worker-comfyui: System info before launch:"
echo "  GPU(s):"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"
echo "  CUDA runtime version:"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null \
    || echo "  (torch not importable)"
echo "  Worker model config:"
echo "    WORKER_MODEL_CONFIG=${WORKER_MODEL_CONFIG:-unset}"
[ -f "${WORKER_MODEL_CONFIG:-/etc/worker/model.yaml}" ] && \
    cat "${WORKER_MODEL_CONFIG:-/etc/worker/model.yaml}" | sed 's/^/    /'
echo "  System RAM:"
free -h 2>/dev/null | head -2 || echo "  (free not available)"
echo ""

echo "worker-comfyui: Starting ComfyUI"

: "${COMFY_LOG_LEVEL:=DEBUG}"

EXTRA_PATHS="--extra-model-paths-config /comfyui/extra_model_paths.yaml"
COMFY_LOG="/var/log/comfyui.log"

COMFY_CMD="python -u /comfyui/main.py --disable-auto-launch --disable-metadata ${EXTRA_PATHS} --verbose ${COMFY_LOG_LEVEL} --log-stdout"
if [ "${SERVE_API_LOCALLY:-}" = "true" ]; then
    COMFY_CMD="${COMFY_CMD} --listen"
fi

# Automatic restart settings (override via environment variables)
: "${COMFY_RESTART_DELAY:=5}"
: "${COMFY_MAX_RAPID_RESTARTS:=5}"
: "${COMFY_RAPID_RESTART_WINDOW:=60}"

comfyui_restart_loop() {
    set -o pipefail
    local rapid_count=0
    local window_start
    window_start=$(date +%s)

    while true; do
        echo "worker-comfyui: Launching ComfyUI process..."
        ${COMFY_CMD} 2>&1 | tee "${COMFY_LOG}"
        local exit_code=$?

        echo "worker-comfyui: ComfyUI exited with code ${exit_code}"

        local now
        now=$(date +%s)
        if (( now - window_start < COMFY_RAPID_RESTART_WINDOW )); then
            rapid_count=$((rapid_count + 1))
        else
            rapid_count=1
            window_start=$now
        fi

        if (( rapid_count >= COMFY_MAX_RAPID_RESTARTS )); then
            echo "worker-comfyui: FATAL — ComfyUI crashed ${rapid_count} times within ${COMFY_RAPID_RESTART_WINDOW}s, not restarting."
            return 1
        fi

        echo "worker-comfyui: Restarting ComfyUI in ${COMFY_RESTART_DELAY}s (crash ${rapid_count}/${COMFY_MAX_RAPID_RESTARTS} in window)..."
        sleep "${COMFY_RESTART_DELAY}"
    done
}

comfyui_restart_loop &
COMFY_LOOP_PID=$!
echo "worker-comfyui: ComfyUI restart loop PID=${COMFY_LOOP_PID}, log=${COMFY_LOG}"

echo "worker-comfyui: Starting RunPod Handler"
if [ "${SERVE_API_LOCALLY:-}" = "true" ]; then
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /handler.py
fi
