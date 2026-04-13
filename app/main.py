"""
app/main.py
-----------
FastAPI application entry point.

CHANGES FOR METRICS:
  - Added GET /metrics endpoint that Prometheus scrapes every 30 seconds
  - Added GET /health endpoint (Kubernetes liveness probe)
  - prometheus_client.REGISTRY is the default global registry; generate_latest()
    serializes all registered metrics into the Prometheus text exposition format

SECURITY NOTE ON /metrics:
  In production, /metrics should NOT be publicly accessible.
  Recommended patterns (in order of preference):
    1. EC2 Security Group: allow port 8000 only from Prometheus server's IP
    2. Separate internal port: run metrics on :9091 (not exposed via ALB)
    3. Mutual TLS between Prometheus and your app
"""

from dotenv import load_dotenv

# Load .env before any other imports so all env vars are available at import time
load_dotenv()

from fastapi import FastAPI                                        # noqa: E402
from fastapi.responses import Response                             # noqa: E402
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST # noqa: E402

from app.middleware.auth import APIKeyMiddleware                   # noqa: E402
from app.routes.query import router as query_router               # noqa: E402

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic RAG Platform",
    description="Production-grade RAG system for financial filing analysis",
    version="1.0.0",
)

# ── Middleware ───────────────────────────────────────────────────────────────
# API key auth is applied globally EXCEPT for /metrics and /health.
# Those endpoints are internal infrastructure — not user-facing.
app.add_middleware(APIKeyMiddleware)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(query_router)


# ── /metrics ─────────────────────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
async def metrics():
    """
    Prometheus scrape endpoint.

    Prometheus hits this URL on its configured scrape_interval (default 15s).
    generate_latest() serializes all metrics registered in the default
    REGISTRY into the Prometheus text exposition format, e.g.:

      # HELP rag_request_latency_seconds End-to-end latency ...
      # TYPE rag_request_latency_seconds histogram
      rag_request_latency_seconds_bucket{intent="revenue_summary",le="13.0"} 4.0
      rag_request_latency_seconds_bucket{intent="revenue_summary",le="15.0"} 6.0
      ...

    include_in_schema=False hides this from the Swagger UI — it is not a
    user-facing API endpoint.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
async def health():
    """
    Kubernetes liveness probe endpoint (minikube deploy).
    Returns 200 OK when the app is running. No auth required.
    Also serves as the Prometheus 'up' metric target — if this endpoint
    stops responding, Prometheus fires an 'InstanceDown' alert.
    """
    return {"status": "ok", "version": "1.0.0"}

