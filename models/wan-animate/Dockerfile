ARG BASE_VERSION
FROM jmendapara/runpod-worker-base:${BASE_VERSION}

# Optional build-time secrets for model_downloads
ARG HUGGINGFACE_ACCESS_TOKEN
ARG CIVITAI_TOKEN

COPY model.yaml /etc/worker/model.yaml
COPY . /tmp/model-ctx/

RUN HUGGINGFACE_ACCESS_TOKEN="${HUGGINGFACE_ACCESS_TOKEN:-}" \
    CIVITAI_TOKEN="${CIVITAI_TOKEN:-}" \
    /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx \
 && rm -rf /tmp/model-ctx /root/.cache/pip /root/.cache/uv && uv cache clean
