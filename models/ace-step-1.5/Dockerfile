ARG BASE_VERSION
FROM jmendapara/runpod-worker-base:${BASE_VERSION}

COPY model.yaml /etc/worker/model.yaml
COPY . /tmp/model-ctx/

RUN /opt/worker/apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx \
 && rm -rf /tmp/model-ctx /root/.cache/pip /root/.cache/uv && uv cache clean
