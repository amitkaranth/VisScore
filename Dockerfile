# GPU image for Google Compute Engine / Vertex AI / self-managed GKE with NVIDIA drivers.
# Build: docker build -t visscore-synthetic .
# Run (example): docker run --gpus all -e HF_TOKEN -v /data/out:/out visscore-synthetic \
#   --out-dir /out --model flux-schnell
#
# Cloud: push to Artifact Registry, run on a GPU VM or custom training job; pass HF_TOKEN via Secret Manager / env.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml setup.py README.md ./
COPY src ./src

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["visscore-generate"]
