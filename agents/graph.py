"""
agents/graph.py
---------------
Wires all agent nodes into a compiled LangGraph StateGraph.

Graph topology:
    START
      |
      v
    query_analyzer          -- extracts tickers[], fiscal_year, intent, decides route
      |
      +-- route == "abstain" --> abstain --> END
      |
      +-- route == "sec" ------> sec_retriever   -- per-ticker candidate pool (pgvector)
                                     |
                                     v
                                  reranker        -- narrow to top-k per ticker;
                                     |               vector score (default) or
                                     v               cross-encoder if RERANK_ENABLED
                                  market_analyst  -- GPT-4o, synthesizes sourced claims
                                     |
                                     v
                                  critic ── PASS/ABSTAIN (final_answer set) ──> END
                                     │
                                     └── RETRY (final_answer None) ──> sec_retriever
                                                                        (max 2 retries)

This is a single orchestrator making routing/retry decisions over specialized
tool nodes — not multiple autonomous agents. See README "Architecture" for
that framing.

Usage:
    from agents.graph import compiled_graph
    result = compiled_graph.invoke({"query": "...", "retry_count": 0})
    print(result["final_answer"])
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.nodes.query_analyzer import query_analyzer_node
from agents.nodes.sec_retriever import sec_retriever_node
from agents.nodes.reranker import rerank_node
from agents.nodes.market_analyst import market_analyst_node
from agents.nodes.critic import critic_node
from agents.nodes.abstain import abstain_node

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def route_after_analysis(state: AgentState) -> str:
    """
    Conditional edge function — called after query_analyzer_node.

    Returns:
        "sec"     -> routes to sec_retriever
        "abstain" -> routes to abstain (no valid ticker / out-of-scope intent /
                     un-ingested fiscal year)
    """
    route = state.get("route") or "abstain"
    logger.info(f"QueryAnalyzer routed -> {route}")
    return route


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function — called after critic_node.

    critic_node itself decides pass vs. abstain and sets final_answer in
    both cases (abstain gets an honest "insufficient evidence" message, not
    a null). Only the retry case leaves final_answer as None. So this is a
    simple two-way branch, not a three-way one — the three-way decision
    already happened inside critic_node.
    """
    if state.get("final_answer") is not None:
        logger.info(f"Critic settled (score={state.get('quality_score', 0):.2f}) -> routing to END")
        return "end"
    logger.info(f"Critic RETRY (retry_count={state.get('retry_count', 0)}) -> routing to sec_retriever")
    return "retry"


def build_graph() -> CompiledStateGraph:
    """Constructs and compiles the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("sec_retriever", sec_retriever_node)
    graph.add_node("reranker", rerank_node)
    graph.add_node("market_analyst", market_analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("abstain", abstain_node)

    graph.add_edge(START, "query_analyzer")

    graph.add_conditional_edges(
        "query_analyzer",
        route_after_analysis,
        {"sec": "sec_retriever", "abstain": "abstain"},
    )

    graph.add_edge("sec_retriever", "reranker")
    graph.add_edge("reranker", "market_analyst")
    graph.add_edge("market_analyst", "critic")
    graph.add_edge("abstain", END)

    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"end": END, "retry": "sec_retriever"},
    )

    return graph.compile()


compiled_graph = build_graph()
