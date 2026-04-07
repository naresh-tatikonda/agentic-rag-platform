"""
app/main.py
-----------
FastAPI application entry point.

Responsibilities:
  1. Initialize the FastAPI app with metadata
  2. Register middleware (auth runs before every request)
  3. Register route handlers (/query, /health, /metrics)

This file is what uvicorn runs:
  uvicorn app.main:app --host 0.0.0.0 --port 8000

Why separate main.py from routes?
  - main.py handles app-level concerns (middleware, startup, config)
  - routes/ handles endpoint-level concerns (request/response logic)
  - This makes it easy to add new route groups (e.g. /ingest, /eval) later
"""

from dotenv import load_dotenv
load_dotenv()   # must run before os.getenv() is called anywhere

from fastapi import FastAPI
from datetime import datetime, timezone

from app.routes.query import router as query_router
from app.middleware.auth import APIKeyMiddleware
from app.models.schemas import HealthResponse

# Initialize FastAPI — metadata populates the Swagger UI at /docs
app = FastAPI(
    title="Agentic RAG Platform",
    description="Multi-agent SEC 10-K financial analysis — LangGraph + pgvector + FastAPI",
    version="1.0.0",
)

# Register auth middleware — applies to ALL routes before handlers run
app.add_middleware(APIKeyMiddleware)

# Register route groups — prefix can be added here e.g. prefix="/v1"
app.include_router(query_router)


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check():
    """
    Kubernetes liveness probe endpoint.
    Returns 200 if the app process is alive.
    No DB check here — DB health belongs in a separate readiness probe.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
    )
