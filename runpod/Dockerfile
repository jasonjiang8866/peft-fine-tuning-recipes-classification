# Build from the repository root. CUDA runtime libraries come from the cu128 wheels.
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/muon
COPY runpod/requirements.txt /opt/muon/runpod/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r runpod/requirements.txt \
    && /opt/venv/bin/python -m pip check
COPY benchmark_muon.py /opt/muon/benchmark_muon.py
COPY runpod /opt/muon/runpod
ARG SOURCE_COMMIT=unknown
ENV MUON_SOURCE_COMMIT=$SOURCE_COMMIT \
    RUNPOD_PYTHON=/opt/venv/bin/python \
    HF_HOME=/workspace/hf-cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RUN_SUITE=smoke \
    RESULTS_ROOT=/workspace/results
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "/opt/muon/runpod/container-start.sh"]
