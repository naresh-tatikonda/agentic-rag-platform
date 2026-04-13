# tests/test_health.py
# ─────────────────────────────────────────────────────
# Smoke test — verifies the FastAPI app boots correctly
# and the /health endpoint returns 200 OK.
#
# Uses TestClient (no real server needed) — runs in CI
# without requiring a running container or database.
# ─────────────────────────────────────────────────────
from fastapi.testclient import TestClient
from app.main import app

# TestClient spins up the app in-process — no docker needed
client = TestClient(app)


def test_health_returns_200():
    """Health endpoint must return 200 — CI liveness check."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_expected_fields():
    """Health response must include status and version fields."""
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
