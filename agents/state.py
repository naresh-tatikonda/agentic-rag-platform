"""
agents/state.py
---------------
Defines the shared state schema for the LangGraph multi-agent pipeline.

Each node in the graph reads from and writes to this state object.
LangGraph passes the state between nodes automatically — no manual
data passing needed between agents.

Flow:
    User Query
        → QueryAnalyzer   (populates: ticker, fiscal_year, intent)
        → SECRetriever    (populates: retrieved_chunks, retrieval_scores)
        → MarketAnalyst   (populates: draft_answer)
        → Critic          (populates: quality_score, final_answer)

Schema note:
    fiscal_year = the fiscal year the 10-K COVERS (e.g. 2023)
                  NOT the date the filing was downloaded or stored.
                  This maps directly to the fiscal_year column in sec_filings table.
"""

from typing_extensions import TypedDict
from typing import Optional, List


class AgentState(TypedDict):
    """
    Single source of truth passed between all agent nodes.
    Every field is Optional except 'query' (required input)
    and 'retry_count' (starts at 0, increments on low-quality answers).
    """

    # ── Input ────────────────────────────────────────────────────────────
    query: str                          # Raw user question e.g. "What are AAPL risks in 2023?"

    # ── Query Analysis (populated by QueryAnalyzerNode) ──────────────────
    ticker: Optional[str]               # Extracted stock ticker e.g. "AAPL"
    fiscal_year: Optional[int]          # Fiscal year the 10-K covers e.g. 2023
                                        # Distinct from filed_date in DB
    intent: Optional[str]               # Classified intent: "risk_analysis" |
                                        # "revenue_summary" | "business_overview" | "general"

    # ── Retrieval (populated by SECRetrieverNode) ─────────────────────────
    retrieved_chunks: Optional[List[str]]     # Top-k text chunks from pgvector
    retrieval_scores: Optional[List[float]]   # Similarity scores for each chunk

    # ── Generation (populated by MarketAnalystNode) ───────────────────────
    draft_answer: Optional[str]         # LLM-generated answer before quality check

    # ── Critic / Quality Gate (populated by CriticNode) ───────────────────
    quality_score: Optional[float]      # Score 0.0–1.0 (threshold: 0.7 to pass)
    retry_count: int                    # Tracks retries to prevent infinite loops
                                        # Max retries = 2 (defined in graph.py)
    final_answer: Optional[str]         # Approved answer returned to user
