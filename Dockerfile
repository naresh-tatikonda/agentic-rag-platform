# =============================================================================
# Dockerfile — agentic-rag-platform
# Builds a production image that serves the FastAPI app via Gunicorn + Uvicorn
# =============================================================================

# --- Stage: Base image ---
# python:3.12-slim is the official Python image stripped of dev tools
# "slim" keeps the image small (~50MB vs ~900MB for the full image)
# Always pin the version — never use "latest" in production (non-reproducible)
FROM python:3.12-slim

# --- Metadata ---
# LABEL is purely informational — shows up in `docker inspect`
# Good practice for team and recruiter visibility
LABEL maintainer="agentic-rag-platform"
LABEL description="FastAPI + LangGraph RAG platform served via Gunicorn/Uvicorn"

# --- System dependencies ---
# curl is needed for the Docker HEALTHCHECK below
# --no-install-recommends keeps image lean (skips optional packages)
# rm -rf /var/lib/apt/lists/* clears apt cache — reduces final image size
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
# All subsequent commands run from /app inside the container
# Also sets the default directory for `docker exec` sessions
WORKDIR /app

# --- Install Python dependencies BEFORE copying app code ---
# This is a critical Docker optimization: layer caching
# If requirements.txt hasn't changed, Docker reuses the cached pip layer
# and skips reinstalling all packages — saves 2-3 minutes on every build
# where only app code changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir tells pip not to store download cache inside the image
# keeps the image smaller

# --- Copy only the code the app needs to run ---
# app/ — FastAPI routes, schemas, config
# agents/ — LangGraph nodes that app/ imports
# We deliberately exclude tests/, src/ingestion/, notebooks/ (see .dockerignore)
COPY app/ ./app/
COPY agents/ ./agents/

# --- Runtime environment variables (structure only, no secret values) ---
# These declare WHAT variables the app expects at runtime
# Actual values are injected via `docker run -e` or GitHub Actions secrets
# NEVER hardcode real values here — this file is committed to git
ENV PYTHONUNBUFFERED=1
# PYTHONUNBUFFERED=1 forces Python stdout/stderr to flush immediately
# Critical for seeing logs in real time inside containers

ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from writing .pyc files inside the container
# Keeps the running container clean

ENV PORT=8000
# Default port — can be overridden at runtime

# --- Non-root user ---
# Running as root inside a container is a security risk
# If the app is compromised, root access = host escape potential
# Best practice: create a dedicated user with no shell and no home dir
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

# --- Expose port ---
# EXPOSE is documentation — tells Docker and humans what port the app uses
# Does NOT actually publish the port (that's `docker run -p 8000:8000`)
EXPOSE 8000

# --- Health check ---
# Docker will call this every 30s to verify the container is healthy
# After 3 failures → container marked "unhealthy"
# Kubernetes and ECS use this to decide whether to restart the container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --- Start command ---
# Gunicorn manages worker processes; each worker is a uvicorn ASGI worker
# -w 3 → 3 workers (formula: 2 × CPU cores + 1, t2.micro has 1 vCPU)
# --timeout 120 → kill workers that hang for >120s (LLM calls can be slow)
# --access-logfile - → stream access logs to stdout (visible in `docker logs`)
CMD ["gunicorn", "app.main:app", \
     "-w", "3", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]
