# syntax=docker/dockerfile:1.7
# TRELLIS.2-4B image-to-3D worker — microsoft/TRELLIS.2 wrapped in a RunPod serverless handler.
#
# NOT a ComfyUI model.yaml worker: the official pipeline runs as-is (trellis2_image_to_3d.py),
# then the handler bakes a PBR GLB (o_voxel.postprocess.to_glb) and renders a turntable MP4.
# Build from the REPO ROOT: `MODEL=trellis2-4b bash build.sh` (build.sh finds
# trainers/<name>/Dockerfile when models/<name> does not exist). The HF token is needed:
# facebook/dinov3-vitl16-pretrain-lvd1689m is gated (accept its licence on the token's account).
#
# Stack: CUDA 12.8 devel (nvcc for the CUDA extensions below; FlexGEMM's Triton kernels need gcc at
# run time too), Python 3.12, torch 2.8 cu128. Upstream recommends torch 2.6 cu124; 2.8 is what the
# extensions and flash-attn ship wheels / compile cleanly for. Every upstream repo is pinned.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
ARG FLEXGEMM_COMMIT=6dd94a859c26ee8246888502eada3dd8ad85532e
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
ARG NVDIFFREC_COMMIT=b296927cc7fd01c2ac1087c8065c4d7248f72da4
ARG UTILS3D_COMMIT=9a4eb15e4021b67b12c460c7057d642626897ec8
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

# Weights (Hugging Face revisions pinned).
ARG TRELLIS2_REV=af44b45f2e35a493886929c6d786e563ec68364d
ARG TRELLIS1_REV=25e0d31ffbebe4b5a97464dd851910efc3002d96
ARG DINOV3_REV=ea8dc2863c51be0a264bab82070e3e8836b02d51
ARG BIREFNET_REV=e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PIP_NO_INPUT=1 \
    PYTHONUNBUFFERED=1 \
    # A100, A6000/3090, L40S/4090/L4, H100/H200. flash-attn's wheel has no Blackwell (sm_120) kernels.
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=8 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=3 \
        python3.12 python3.12-venv python3.12-dev \
        build-essential ninja-build git ca-certificates curl \
        libgl1 libglib2.0-0 libjpeg-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel
ENV PATH="/opt/venv/bin:${PATH}"

# Torch stack first (largest, rarely changes).
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url ${PYTORCH_INDEX_URL}

# TRELLIS.2's --basic deps (setup.sh) minus gradio/tensorboard/pandas/lpips (demo + training only),
# BiRefNet's remote-code deps, and the worker's own.
COPY trainers/trellis2-4b/requirements.txt /tmp/worker-requirements.txt
RUN pip install --no-cache-dir -r /tmp/worker-requirements.txt && \
    pip install --no-cache-dir "git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_COMMIT}" && \
    rm /tmp/worker-requirements.txt

# flash-attn 2.8.3, the official wheel for torch 2.8 / CUDA 12 / cp312 (no compile).
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Fetch one exact commit of a repo (with submodules) into /tmp/ext/<name>.
RUN printf '%s\n' '#!/bin/sh' 'set -e' \
      'mkdir -p /tmp/ext && cd /tmp/ext && git init -q "$1" && cd "$1"' \
      'git remote add origin "$2" && git fetch -q --depth 1 origin "$3" && git checkout -q FETCH_HEAD' \
      'git submodule update -q --init --recursive --depth 1' \
      > /usr/local/bin/fetch-commit && chmod +x /usr/local/bin/fetch-commit

# CUDA extensions, compiled for TORCH_CUDA_ARCH_LIST (no GPU needed at build time).
RUN fetch-commit nvdiffrast https://github.com/NVlabs/nvdiffrast.git ${NVDIFFRAST_COMMIT} && \
    pip install --no-cache-dir /tmp/ext/nvdiffrast --no-build-isolation && rm -rf /tmp/ext/nvdiffrast
RUN fetch-commit nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git ${NVDIFFREC_COMMIT} && \
    pip install --no-cache-dir /tmp/ext/nvdiffrec --no-build-isolation && rm -rf /tmp/ext/nvdiffrec
RUN fetch-commit CuMesh https://github.com/JeffreyXiang/CuMesh.git ${CUMESH_COMMIT} && \
    pip install --no-cache-dir /tmp/ext/CuMesh --no-build-isolation && rm -rf /tmp/ext/CuMesh
RUN fetch-commit FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git ${FLEXGEMM_COMMIT} && \
    pip install --no-cache-dir /tmp/ext/FlexGEMM --no-build-isolation && rm -rf /tmp/ext/FlexGEMM

# TRELLIS.2 itself (the trellis2 package is used from the checkout) + its o-voxel extension.
RUN fetch-commit TRELLIS.2 https://github.com/microsoft/TRELLIS.2.git ${TRELLIS2_COMMIT} && \
    mv /tmp/ext/TRELLIS.2 /app && cd /app && \
    pip install --no-cache-dir ./o-voxel --no-build-isolation && \
    rm -rf .git o-voxel/third_party data_toolkit configs train.py && \
    # Installed + compiled, checked without importing: FlexGEMM reads the GPU name at import time and
    # the build box has no GPU (the handler's cold start imports everything on the real worker).
    python -c "import importlib.util as u, sys; mods=['o_voxel','cumesh','flex_gemm','nvdiffrast','nvdiffrec_render','flash_attn']; missing=[m for m in mods if u.find_spec(m) is None]; sys.exit('missing: %s' % missing) if missing else print('extensions installed OK')"

# Weights, one layer per repo (each blob stays inside Docker Hub's upload window).
# Only the checkpoints the image-to-3D pipeline loads; the VAE encoders (texturing / training) are skipped.
ENV HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_TELEMETRY=1
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("microsoft/TRELLIS.2-4B", revision="${TRELLIS2_REV}", local_dir="/models/TRELLIS.2-4B",
                  allow_patterns=["pipeline.json", "ckpts/ss_flow_*", "ckpts/shape_dec_*", "ckpts/tex_dec_*", "ckpts/slat_flow_*"])
PY
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("microsoft/TRELLIS-image-large", revision="${TRELLIS1_REV}", local_dir="/models/TRELLIS-image-large",
                  allow_patterns=["ckpts/ss_dec_conv3d_16l8_fp16.*"])
snapshot_download("ZhengPeng7/BiRefNet", revision="${BIREFNET_REV}", local_dir="/models/BiRefNet")
PY
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("facebook/dinov3-vitl16-pretrain-lvd1689m", revision="${DINOV3_REV}", local_dir="/models/dinov3-vitl16")
PY

# Point pipeline.json at the local copies. Background removal: ZhengPeng7/BiRefNet (MIT, the code's own
# default) instead of briaai/RMBG-2.0 (CC BY-NC, gated), which the published pipeline.json names.
RUN python - <<'PY'
import json
p = "/models/TRELLIS.2-4B/pipeline.json"
cfg = json.load(open(p))
a = cfg["args"]
a["models"]["sparse_structure_decoder"] = "/models/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
assert a["image_cond_model"]["name"] == "DinoV3FeatureExtractor"
a["image_cond_model"]["args"]["model_name"] = "/models/dinov3-vitl16"
assert a["rembg_model"]["name"] == "BiRefNet"
a["rembg_model"]["args"]["model_name"] = "/models/BiRefNet"
json.dump(cfg, open(p, "w"), indent=4)
print(json.dumps({k: a[k] for k in ("models", "image_cond_model", "rembg_model", "default_pipeline_type")}, indent=2))
PY

COPY trainers/trellis2-4b/*.py /app/worker/

WORKDIR /app
ENV PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ATTN_BACKEND=flash_attn \
    TOKENIZERS_PARALLELISM=false

CMD ["python", "-u", "/app/worker/handler.py"]
