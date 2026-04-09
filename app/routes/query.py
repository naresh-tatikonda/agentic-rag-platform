"""
app/routes/query.py
-------------------
Defines POST /query — the main entry point for the agentic RAG system.

This route is intentionally thin:
  - Validate input     → Pydantic handles this automatically
  - Build initial state → map request fields to AgentState shape
  - Run the graph      → delegate everything to LangGraph
  - Return response    → map AgentState output to QueryResponse

Why keep routes thin?
  - Business logic lives in the graph nodes, not in HTTP handlers
  - Easier to test graph logic independently of FastAPI
  - Clean separation: HTTP layer vs. agent layer
"""

import time
from datetime import datetime, timezone
from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.logger import get_logger, log_query_request      
from agents.graph import compiled_graph
# Import from central metrics module — never instantiate metrics here directly
# as that causes duplicate registration errors on hot reload.
from app.metrics import (
    RAG_LATENCY,
    RAG_QUALITY_SCORE,
    RAG_RETRY_TOTAL,
    RAG_REQUESTS_TOTAL,
)

router = APIRouter()
logger = get_logger("query")


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def run_query(request: QueryRequest):
    """
    Runs the full 4-node LangGraph pipeline:
    QueryAnalyzer → SECRetriever → MarketAnalyst → Critic

    If Critic quality_score < 0.7, graph retries up to MAX_RETRIES=2.
    Returns final approved answer with full quality metadata.
    """

    # Capture start time before graph runs — used to compute latency
    start_time = time.perf_counter()
    error_message = None

    intent = "unknown"   # populated by QueryAnalyzerNode inside the graph
    status = "error"     # updated to "success" if graph completes cleanly

    # Build initial AgentState — only populate input fields here
    # Each agent node populates its own fields as the graph runs
    try:
        initial_state = {
            "query": request.query,
            "ticker": request.ticker,
            "fiscal_year": request.fiscal_year,
            "retry_count": 0,          # always start at 0
        }

        # Invoke compiled LangGraph graph — this runs all 4 nodes sequentially
        # with conditional retry edge from Critic back to SECRetriever
        result = compiled_graph.invoke(initial_state)

        # Calculate total end-to-end latency in milliseconds
        latency_ms = (time.perf_counter() - start_time) * 1000

         # UNPACK RESULT — extract real values from AgentState
        intent        = result.get("intent", "unknown")
        quality_score = result.get("quality_score", 0.0)
        retry_count   = result.get("retry_count", 0)
        latency_s     = latency_ms / 1000

        # RECORD METRICS — after unpack, before return
        RAG_LATENCY.labels(intent=intent).observe(latency_s)
        RAG_QUALITY_SCORE.labels(intent=intent).set(quality_score)
        if retry_count > 0:
            RAG_RETRY_TOTAL.labels(intent=intent).inc(retry_count)
        RAG_REQUESTS_TOTAL.labels(intent=intent, status="success").inc()


        # Log successful request
        log_query_request(
            logger=logger,
            query=request.query,
            ticker=result.get("ticker", request.ticker),
            fiscal_year=result.get("fiscal_year", request.fiscal_year),
            intent=result.get("intent"),
            quality_score=result.get("quality_score", 0.0),
            retry_count=result.get("retry_count", 0),
            latency_ms=latency_ms,
        )    

        # Map AgentState output fields to QueryResponse Pydantic model
        return QueryResponse(
            final_answer=result.get("final_answer", "No answer generated"),
            ticker=result.get("ticker", request.ticker),
            fiscal_year=result.get("fiscal_year", request.fiscal_year),
            intent=result.get("intent"),
            quality_score=result.get("quality_score", 0.0),
            retrieval_scores=result.get("retrieval_scores", []),
            retry_count=result.get("retry_count", 0),
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now(timezone.utc),
        )

    except Exception as e:                          
        latency_ms = (time.perf_counter() - start_time) * 1000
        error_message = str(e)

        # ERROR METRICS — uses defaults from step 1 if graph never ran
        RAG_LATENCY.labels(intent=intent).observe(latency_s)
        RAG_REQUESTS_TOTAL.labels(intent=intent, status="error").inc()

        # Log failed request
        log_query_request(
            logger=logger,
            query=request.query,
            ticker=request.ticker,
            fiscal_year=request.fiscal_year,
            intent=None,
            quality_score=0.0,
            retry_count=0,
            latency_ms=latency_ms,
            error=error_message,
        )
        raise   # re-raise so FastAPI returns 500
