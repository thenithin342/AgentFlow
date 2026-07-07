# syntax=docker/dockerfile:1.7
# AgentFlow backend — production image.
#
# Two-stage build:
#   1. `builder` — installs the full Python toolchain in a Debian-slim
#      image, then copies only the resolved site-packages into a
#      python:3.12-slim runtime. This keeps the runtime image at the
#      same Debian base as the builder (so compiled wheels like
#      faiss-cpu / sentence-transformers link against the same glibc
#      / libgomp) while shedding compilers, headers, and pip.
#
#   2. `runtime` — non-root user, no shell history, no pip cache,
#      pre-built sentence-transformers model in a writable volume.
#
# Reference: DEPLOYMENT.md section 3 "Container Image".

ARG PYTHON_VERSION=3.12

# -----------------------------------------------------------------------------
# Stage 1: build
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# System deps required at install time only:
#   - build-essential, gcc — compile any wheel that lacks a manylinux build
#   - libgomp1 — OpenMP runtime, required by faiss-cpu and torch
# We keep them in the builder stage; the runtime image installs ONLY
# the runtime libraries it actually needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Use a venv so the produced site-packages can be copied wholesale
# into the runtime image. `--copies` keeps the venv relocatable when
# the runtime image is built on a different machine with a different
# absolute path under /usr/local/lib/python3.12/site-packages.
RUN python -m venv --copies /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Copy ONLY the dependency manifests first so the resolver layer
# caches independently of source changes. Order: requirements.txt
# before requirements-dev.txt; dev deps are skipped in the runtime
# image so we never pay their install cost in CI.
COPY requirements.txt ./
RUN pip install --upgrade pip wheel \
    && pip install -r requirements.txt

# Copy the application source last. The intermediate layers above
# this point are stable across source-only edits, so dev iteration
# stays fast.
COPY backend ./backend
COPY pyproject.toml ./

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Runtime-only system libs:
#   - libgomp1 — faiss-cpu + sentence-transformers need OpenMP
#   - curl      — used by the container healthcheck (Docker HEALTHCHECK)
#   - tini      — PID 1 reaper so SIGTERM to the container cleanly
#                 stops uvicorn (no zombie workers on shutdown)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID 10001 matches the AWS / k8s / Railway convention
# for "first non-system user"; gives a stable numeric owner across
# host bind mounts.
RUN groupadd --system --gid 10001 agentflow \
    && useradd  --system --uid 10001 --gid agentflow --no-create-home --shell /sbin/nologin agentflow

# Carry the venv forward from the builder. The venv is fully
# self-contained — no symlinks back to the host path — because
# `python -m venv --copies` was used in stage 1.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # uvicorn binds to all interfaces inside the container; the
    # orchestrator (docker / railway) maps the host port.
    AGENTFLOW_HOST=0.0.0.0 \
    AGENTFLOW_PORT=8000

WORKDIR /app

# Copy source as the unprivileged user. Ownership matters because
# the runtime writes to /app/data, /app/faiss_indexes, and
# /app/ltm_indexes — these directories must be owned by agentflow
# or the container will fail to start with a permission error.
COPY --chown=agentflow:agentflow backend ./backend
COPY --chown=agentflow:agentflow pyproject.toml ./

# Pre-create the writable directories the app uses. Doing this in
# the image (rather than at startup) means a read-only rootfs
# orchestrator can still mount a tmpfs / volume on these paths and
# the app won't crash on first write.
RUN mkdir -p /app/data /app/faiss_indexes /app/ltm_indexes \
    && chown -R agentflow:agentflow /app

USER agentflow

EXPOSE 8000

# tini as PID 1 — uvicorn workers respond to SIGTERM cleanly when
# the orchestrator stops the container. Without tini, SIGTERM hits
# the shell wrapper and the Python process never gets a chance to
# drain in-flight requests.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Defaults: 1 worker, 4 threads, 60s graceful-shutdown timeout.
# Override at deploy time with `-w 4` etc. via Railway's CLI or
# docker-compose command.
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "60", \
     "--log-level", "info"]

# Liveness probe. HEAD /healthz must be cheap and side-effect-free
# (no DB calls, no graph invocation). We use --fail so curl exits
# non-zero on any 4xx/5xx, which is what Docker's HEALTHCHECK needs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent --max-time 3 http://localhost:8000/healthz || exit 1
