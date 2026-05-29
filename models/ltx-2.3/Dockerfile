# syntax=docker/dockerfile:1.7
ARG BASE_VERSION
FROM jmendapara/runpod-worker-base:${BASE_VERSION}

COPY model.yaml /etc/worker/model.yaml
COPY . /tmp/model-ctx/

# ltx-2.3 bakes ~130 GB of weights. A single RUN producing one monolithic
# layer cannot be pushed to Docker Hub (one ~150 GB blob exceeds the upload
# session window → "blob upload invalid - upload state expired"). So the
# weight downloads are SHARDED across separate RUN steps, each yielding its
# own image layer. Per-shard blobs fit the session window and failed pushes
# resume blob-by-blob. Shard K only fetches model_downloads where
# (index % 8 == K); large checkpoints land in distinct shards.
#
# Secrets (HF + Civitai tokens) are mounted per-RUN, never baked into layers.

# --- setup: custom nodes + pip (small layer) ---
RUN --mount=type=secret,id=hf_token \
    --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase setup

# --- weight shards: each RUN = one independently-pushable layer ---
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard 0/8
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard 1/8
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard 2/8
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard 3/8
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard 4/8
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard 5/8
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard 6/8
RUN --mount=type=secret,id=hf_token --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase download --shard 7/8

# --- finalize: post_install + env, then drop build context (small layer) ---
RUN --mount=type=secret,id=hf_token \
    --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx --phase finalize \
 && rm -rf /tmp/model-ctx
