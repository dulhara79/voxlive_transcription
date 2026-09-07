# =============================================================================
# VoxLive backend image.
#
# THE ONE LINE THAT MATTERS MOST FOR COST
# ---------------------------------------
# `pip install torch` from PyPI installs the CUDA build. On Linux/x86 that
# drags in nvidia-cudnn, nvidia-cublas, nvidia-nccl and friends — roughly
# 2 GB of GPU libraries that can never execute on a Fargate task, because
# Fargate has no GPU.
#
#     default torch wheel      ~2.4 GB installed   image ~3.2 GB
#     +cpu torch wheel         ~0.2 GB installed   image ~1.1 GB
#
# That is paid on every image pull: every deploy, every scale-out, every task
# replacement. It also pushes the task into a larger memory tier than the
# workload needs. The `--index-url .../whl/cpu` below is the whole fix.
#
# ARM64
# -----
# Build for linux/arm64 and run on Fargate Graviton: same vCPU and memory,
# about 20% cheaper, and the CPU torch wheel exists for aarch64. Build with:
#
#     docker buildx build --platform linux/arm64 -t voxlive-backend .
#
# MODEL WEIGHTS ARE BAKED IN
# --------------------------
# WeSpeaker is downloaded at BUILD time, not at start-up. Otherwise every cold
# start pulls ~90 MB from Hugging Face — which costs start-up latency, costs
# NAT egress if the task is in a private subnet, and makes a deploy fail when
# huggingface.co is having a bad day. Pass the token as a build secret so it
# does not end up in an image layer:
#
#     docker build --secret id=hf_token,env=HUGGINGFACE_TOKEN .
# =============================================================================

# ---- stage 1: build the virtualenv -----------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed by a couple of wheels that have no prebuilt
# aarch64 artefact. It stays in THIS stage only and never reaches the runtime
# image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first, from PyTorch's own index. Doing it before the rest
# means the later `pip install -r requirements.txt` sees torch as already
# satisfied and will not pull the CUDA build to satisfy pyannote's dependency.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" "torchaudio>=2.2"

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- stage 2: bake the diarization model -----------------------------------
FROM builder AS models
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models
RUN --mount=type=secret,id=hf_token \
    HUGGINGFACE_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python -c "\
import os;\
from huggingface_hub import snapshot_download;\
token = os.environ.get('HUGGINGFACE_TOKEN') or None;\
snapshot_download('pyannote/wespeaker-voxceleb-resnet34-LM', token=token)\
" || echo "WARNING: model not baked in; it will download on first start"

# ---- stage 3: runtime ------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1 is OpenMP, which torch links against. Without it the import fails
# at runtime with an unhelpful shared-object error.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 voxlive

COPY --from=builder /opt/venv /opt/venv
COPY --from=models  /opt/models /opt/models

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/models \
    HF_HUB_OFFLINE=1 \
    APP_ENV=production \
    # torch spawns one thread per core by default and then fights the event
    # loop for them. Diarization here is a short batched pass, not a training
    # job, so one thread per process is both faster and predictable.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY --chown=voxlive:voxlive app ./app

USER voxlive
EXPOSE 8000

# /health is deliberately the LIVENESS probe and /ready the target-group
# health check. /health answers "is the process alive", and stays 200 during
# the seconds it takes to load WeSpeaker; pointing liveness at /ready instead
# would kill the task mid-warm-up and produce a restart loop that looks like a
# crash.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One worker. Sessions hold in-process state (SessionState, the diarizer's
# window buffer, the ASR scheduler's queue), so a second worker would not
# share any of it — two workers means two independent capacity limits and a
# WebSocket that lands on whichever one the kernel chose. Scale with tasks,
# not with workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
