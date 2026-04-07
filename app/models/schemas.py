"""
app/models/schemas.py
---------------------
Pydantic models for API request and response validation.

FASTAPI uses these models to:
    1. Automatically validate incoming request data (wrong type = 422 error)
    2. Serializes outgoing response data to JSON
    3. Auto-generate Swagger UI at /docs

These models mirror AgentState fields - request maps to input fields,
response maps to fields computed by each agent node.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class QueryRequest(BaseModel):
    """
    Incoming request for body Post /query.
    Only 3 fields needed - the agent extracts everything else from the question.
    """

    query: str = Field(
            ...,
            description="Natural language question about SEC filings",
            example="What are the Apple's main risk factors in FY2023"
     )
    ticker: Optional[str] = Field(
        default="AAPL",
        description="Stock ticker to scope retrieval. Defaults to AAPL.",
        example="AAPL"
    )
    fiscal_year: Optional[int] = Field(
        default=2023,
        description="Fiscal year the 10-K covers. Defaults to 2023.",
        example=2023
    )

class QueryResponse(BaseModel):
    """
    Response body for POST /query.
    Include answer + quality metadata that proves production-grade self-evaluation.
    Every field maps to a field written by a specific agent node in AgentState.
    """
    final_answer: str = Field(
            description="Approved answer from MarketAnalyst, passed by critic quality gate"
    )
    ticker: str = Field(
            description="Ticker used for retrieval scoping"
    )
    fiscal_year: int = Field(
            description="Fiscal year used for retrieval scoping"
    )
    intent: Optional[str] = Field(
            description="Query intent extracted by QueryAnalyzer e.g. risk_analysis"
    )
    quality_score: float = Field(
            description="Critic quality score 0.0-1.0. Must exceed 0.7 to return answer."
    )
    retrieval_scores: Optional[List[float]] = Field(
        description="Cosine similarity scores for each retrieved chunk from pgvector"
    )
    retry_count: int = Field(
        description="Number of agent retries before final answer was approved"
    )
    latency_ms: float = Field(
        description="End-to-end query latency in milliseconds"
    )
    timestamp: datetime = Field(
        description="UTC timestamp of the response"
    )

class HealthResponse(BaseModel):
    """
    Response body for GET /health.
    Used by kubernetes liveness probe to verify the pod is alive.
    kept intentionally simple - no DB checks here (that's a readiness probe).
    """
    status: str = Field(description="'healthy' if app is running")
    version: str = Field(description="App version running")
    timestamp: datetime = Field(description="UTC timestamp of the health check")







        


