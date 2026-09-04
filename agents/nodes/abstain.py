"""
agents/nodes/abstain.py
------------------------
Abstain Node — reached directly from QueryAnalyzer when route == "abstain".

Responsibility:
    Produce an honest "can't answer this" response and skip retrieval and
    generation entirely. No LLM call, no chunks, nothing to hallucinate from.

Why a dedicated node instead of falling through the SEC pipeline?
    Running SECRetriever/MarketAnalyst on a query with no valid ticker, an
    out-of-scope intent (e.g. price performance), or an un-ingested fiscal
    year wastes the retrieval + generation calls on a query that was never
    going to be answerable, and risks the LLM improvising an answer anyway
    despite empty/irrelevant context. Short-circuiting here is both cheaper
    and safer.

Output:
    Updates AgentState with: final_answer, quality_score=0.0
"""

import logging

from agents.state import AgentState

logger = logging.getLogger(__name__)


def abstain_node(state: AgentState) -> AgentState:
    reason = state.get("abstain_reason") or "this query is outside what the system can answer"
    logger.info(f"Abstain: {reason}")
    return {
        "final_answer": f"I can't answer that from the ingested SEC filings: {reason}.",
        "quality_score": 0.0,
    }
