# syntax=docker/dockerfile:1.7
# LoRA trainer worker — ostris/ai-toolkit wrapped in a RunPod serverless handler.
#
# NOT a ComfyUI model.yaml worker: no weights are baked. Base models are pulled from
# Hugging Face on first use into HF_HOME, which lives on the endpoint's network volume
# (/runpod-volume/hf) when one is attached, else on the container disk (/workspace/hf).
# Build from the REPO ROOT: `MODEL=lora-trainer bash build.sh` (build.sh finds
# trainers/<name>/Dockerfile when models/<name> does not exist).
#
# Stack mirrors ai-toolkit's own docker/Dockerfile (CUDA 13.0.3 runtime + torch 2.13 cu130,
# python 3.12) minus the web UI. gcc/libc6-dev: Triton JIT-compiles kernel launchers at
# runtime (int8 convrot / nvfp4 quantized models fail with "Failed to find C compiler").
ARG BASE_IMAGE=nvidia/cuda:13.0.3-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG AI_TOOLKIT_COMMIT=e9c72b762748e4d31837fbeb3b87d01a5b143321
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PIP_NO_INPUT=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0" \
    AI_TOOLKIT_DIR=/app/ai-toolkit \
    AI_TOOLKIT_COMMIT=${AI_TOOLKIT_COMMIT}

RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=3 \
        python3.12 python3.12-venv python3.12-dev \
        gcc libc6-dev \
        git git-lfs wget curl ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libsndfile1 ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel
ENV PATH="/opt/venv/bin:${PATH}"

# Torch stack first (largest, rarely changes).
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url ${PYTORCH_INDEX_URL}

# ai-toolkit at a pinned commit (fetch exactly that SHA, no history).
RUN mkdir -p /app && cd /app && git init -q ai-toolkit && cd ai-toolkit && \
    git remote add origin https://github.com/ostris/ai-toolkit.git && \
    git fetch -q --depth 1 origin ${AI_TOOLKIT_COMMIT} && \
    git checkout -q FETCH_HEAD && rm -rf .git

WORKDIR /app/ai-toolkit
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir setuptools==69.5.1

# Attention kernels ai-toolkit's own image ships (prebuilt wheels, no compiler at build time).
RUN pip install --no-cache-dir -U \
        torchcodec==0.15.0 \
        natten==0.21.7+torch2130cu130 --find-links https://whl.natten.org \
        https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
    && pip install --no-cache-dir flash-linear-attention==0.5.2 \
    && rm -rf /root/.cache/pip

# Worker runtime deps + the handler.
# Captions are cached 16 at a time by default, which runs Wan 2.2 14B out of memory on an 80 GB card once a
# dataset reaches about 16 files. One at a time costs seconds. The grep fails the build if a bump moves the line.
RUN sed -i 's/^TEXT_EMBED_CACHE_BATCH = 16/TEXT_EMBED_CACHE_BATCH = 1/' toolkit/dataloader_mixins.py && \
    grep -q '^TEXT_EMBED_CACHE_BATCH = 1 ' toolkit/dataloader_mixins.py

COPY trainers/lora-trainer/requirements.txt /tmp/trainer-requirements.txt
RUN pip install --no-cache-dir -r /tmp/trainer-requirements.txt && rm /tmp/trainer-requirements.txt
COPY trainers/lora-trainer/*.py /app/trainer/

WORKDIR /app/trainer
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    TQDM_MININTERVAL=2

CMD ["python", "-u", "/app/trainer/handler.py"]
