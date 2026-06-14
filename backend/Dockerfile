# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH="/opt/venv"

RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

# libgomp1 + libglib2.0-0 cover torch/opencv-headless runtime deps; curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU torch wheels from the PyTorch CPU index first so requirements.lock (which
# pins the same torch version) is already satisfied and never pulls a CUDA build.
RUN pip install --upgrade pip && \
    pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.lock /app/requirements.lock
RUN pip install -r requirements.lock

# App source, including the locked checkpoint (public; committed).
COPY . /app

# Build-time checkpoint SHA-256 gate — fail the build on mismatch (defence in depth;
# the app also asserts this at startup).
RUN python - <<'PY'
import hashlib, sys
from trailscope import config
h = hashlib.sha256()
with open(config.CHECKPOINT_PATH, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
actual = h.hexdigest()
if actual != config.EXPECTED_CHECKPOINT_SHA256:
    sys.exit(f"Checkpoint SHA mismatch at build: {actual} != {config.EXPECTED_CHECKPOINT_SHA256}")
print("Checkpoint SHA verified at build:", actual)
PY

EXPOSE 8000

RUN useradd -m app && chown -R app:app /app
USER app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
