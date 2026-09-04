"""
agents/state.py
---------------
Defines the shared state schema for the LangGraph agent pipeline.

Each node in the graph reads from and writes to this state object.
LangGraph passes the state between nodes automatically — no manual
data passing needed between agents.

Flow:
    User Query
        -> QueryAnalyzer   (populates: tickers, fiscal_year, intent, route)
        -> [route == "sec"]
             -> SECRetriever  (populates: retrieved_chunks, retrieval_scores)
             -> Reranker      (re-scores/trims retrieved_chunks per ticker)
             -> MarketAnalyst (populates: draft_claims, draft_answer)
             -> Critic        (populates: quality_score, ungrounded_claims, final_answer)
        -> [route == "abstain"]
             -> Abstain (populates: final_answer directly, no retrieval/generation)

Schema note:
    fiscal_year = the fiscal year the 10-K COVERS (e.g. 2023)
                  NOT the date the filing was downloaded or stored.
                  This maps directly to the fiscal_year column in sec_filings table.

Retrieved chunk shape (list[dict] in retrieved_chunks):
    {
        "chunk_id": str,        # f"{ticker}:{row_id}" — stable reference for citations
        "ticker": str,
        "text": str,
        "vector_score": float,
        "rerank_score": float | None,  # set by Reranker, None before it runs
    }

Claim shape (list[dict] in draft_claims / ungrounded_claims):
    {"text": str, "source_chunk_id": str, "grounded": bool | None}
"""

from typing_extensions import TypedDict
from typing import Optional, Annotated


def keep_last(existing, new):
    """Reducer: keep new value if set, otherwise keep existing."""
    return new if new is not None else existing


class AgentState(TypedDict):
    """
    Single source of truth passed between all agent nodes.
    Every field is Optional except 'query' (required input)
    and 'retry_count' (starts at 0, increments on low-quality answers).
    """

    # -- Input ---------------------------------------------------------------
    query: str                          # Raw user question e.g. "Compare AMD and AVGO risk factors"

    # -- Query Analysis (populated by QueryAnalyzerNode) ----------------------
    tickers: Annotated[Optional[list], keep_last]              # Extracted tickers e.g. ["AMD", "AVGO"]
    fiscal_year: Annotated[Optional[int], keep_last]            # Fiscal year the 10-K covers e.g. 2025
    intent: Annotated[Optional[str], keep_last]                 # "risk_analysis" | "revenue_summary" |
                                        # "business_overview" | "price_performance" | "comparison" | "general"
    route: Annotated[Optional[str], keep_last]                  # "sec" | "abstain"
    abstain_reason: Annotated[Optional[str], keep_last]         # human-readable reason when route == "abstain"

    # -- Retrieval (populated by SECRetrieverNode, trimmed by RerankerNode) ---
    retrieved_chunks: list      # list[dict] — see module docstring for shape
    retrieval_scores: list      # kept for backward-compat API surface (RAGAS eval) — mirrors
                                 # [c["rerank_score"] or c["vector_score"] for c in retrieved_chunks]

    # -- Generation (populated by MarketAnalystNode) ---------------------------
    draft_claims: Annotated[Optional[list], keep_last]          # list[dict] — claim/source_chunk_id pairs
    draft_answer: Annotated[Optional[str], keep_last]           # claims composed into prose, pre quality-gate

    # -- Critic / Quality Gate (populated by CriticNode) -----------------------
    quality_score: Annotated[Optional[float], keep_last]        # derived from grounded-claim fraction
    ungrounded_claims: Annotated[Optional[list], keep_last]     # claims that failed the grounding check
    retry_count: Annotated[Optional[int], keep_last]            # tracks retries to prevent infinite loops
                                        # Max retries = 2 (defined in graph.py)
    final_answer: Annotated[Optional[str], keep_last]           # approved answer OR abstention message
