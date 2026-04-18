# CPU-friendly image; same CLI as local. For GPU hosts, use NVIDIA base + nvidia-docker.
FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml setup.py README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip setuptools wheel     && pip install --no-cache-dir -e .
ENTRYPOINT ["visscore-generate"]
