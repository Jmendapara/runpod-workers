# syntax=docker/dockerfile:1.7
ARG BASE_VERSION
FROM jmendapara/runpod-worker-base:${BASE_VERSION}

COPY model.yaml /etc/worker/model.yaml
COPY . /tmp/model-ctx/

# Secrets (HF + Civitai tokens) are mounted only for this RUN — never baked
# into image layers. Passed by build.sh via `--secret id=...,env=...`.
RUN --mount=type=secret,id=hf_token \
    --mount=type=secret,id=civitai_token \
    HUGGINGFACE_ACCESS_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    CIVITAI_TOKEN="$(cat /run/secrets/civitai_token 2>/dev/null || true)" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx \
 && rm -rf /tmp/model-ctx /root/.cache/pip /root/.cache/uv && uv cache clean
