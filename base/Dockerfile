# Build from REPO ROOT context with `-f base/Dockerfile` so the schema is reachable.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY=12.8
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PIP_NO_INPUT=1

# System deps. --no-install-recommends keeps the image lean and sidesteps Ubuntu
# mirror flakes for unneeded packages. libsndfile1 + ffmpeg are needed by the
# audio/video collectors (FLAC->WAV transcode, VHS sidecar post-processing).
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=3 \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI via comfy-cli, then force-reinstall PyTorch from the cu128
# index. The reinstall is ~3 GB; doing it once in the base means model images
# never pay this cost.
RUN /usr/bin/yes | comfy --workspace /comfyui install \
        --version "${COMFYUI_VERSION}" \
        --cuda-version "${CUDA_VERSION_FOR_COMFY}" \
        --nvidia && \
    uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL} && \
    rm -rf /root/.cache/pip /root/.cache/uv /root/.cache/comfy-cli /tmp/* && \
    uv cache clean

WORKDIR /comfyui
# comfy-cli's `install` skips parts of ComfyUI's own requirements.txt (e.g.
# sqlalchemy/alembic/aiosqlite for the asset DB), causing crashes. Install
# explicitly.
RUN if [ -f /comfyui/requirements.txt ]; then \
        /opt/venv/bin/pip install -q --root-user-action=ignore -r /comfyui/requirements.txt; \
    fi && \
    rm -rf /root/.cache/pip /root/.cache/uv && \
    uv cache clean

WORKDIR /

COPY base/requirements.txt /tmp/base-requirements.txt
RUN uv pip install -r /tmp/base-requirements.txt && \
    rm /tmp/base-requirements.txt && \
    rm -rf /root/.cache/pip /root/.cache/uv && \
    uv cache clean

COPY base/runtime/extra_model_paths.yaml /comfyui/extra_model_paths.yaml

COPY base/runtime/start.sh /start.sh
COPY base/runtime/handler.py /handler.py
COPY base/runtime/network_volume.py /network_volume.py
COPY base/runtime/apply_model_config.py /opt/worker/apply_model_config.py
COPY base/lib /opt/worker/lib
COPY schema/model.schema.json /opt/worker/schema/model.schema.json
RUN chmod +x /start.sh /opt/worker/apply_model_config.py

COPY base/scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
COPY base/scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-node-install /usr/local/bin/comfy-manager-set-mode

ENV PYTHONPATH="/opt/worker:${PYTHONPATH:-}"
ENV WORKER_MODEL_CONFIG=/etc/worker/model.yaml

CMD ["/start.sh"]
