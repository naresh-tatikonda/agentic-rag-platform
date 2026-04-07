"""
app/middleware/auth.py
----------------------
API key authentication middleware.

Every request must include a valid API key in the X-API-Key header.
Middleware runs BEFORE the request reaches any route handler.

Why middleware instead of FastAPI Depends()?
  - Middleware applies globally to ALL routes automatically
  - Depends() requires adding it manually to each route function
  - For auth, you want global enforcement — middleware is the right tool

Why exempt /health?
  - Kubernetes liveness probes call /health from inside the cluster
  - They don't carry API keys — if we block them, K8s thinks the pod is dead
  - and will restart it unnecessarily
"""

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# Paths that bypass API key check
# /docs, /openapi.json, /redoc → Swagger UI needs these to load
# /health → Kubernetes liveness probe must be open
EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that validates X-API-Key header on every request.
    Inherits from BaseHTTPMiddleware — FastAPI is built on Starlette.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip auth check for exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # Extract API key from request header
        api_key = request.headers.get("X-API-Key")

        # Load expected key from environment — never hardcode secrets
        expected_key = os.getenv("API_KEY")

        # Reject request if key is missing or doesn't match
        if not api_key or api_key != expected_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Pass X-API-Key header."}
            )

        # Key is valid — forward request to route handler
        return await call_next(request)
