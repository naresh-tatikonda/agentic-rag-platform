"""
agents/graph.py
---------------
Wires all agent nodes into a compiled LangGraph StateGraph.

Graph topology:
    START
      ↓
    query_analyzer          — extracts ticker, fiscal_year, intent
      ↓
    sec_retriever           — hybrid pgvector search
      ↓
    market_analyst          — GPT-4o synthesis
      ↓
    critic ─── PASS ──────→ END
      │
      └── RETRY ──────────→ sec_retriever (max 2 retries)

Conditional routing:
    After critic runs, should_retry() inspects state:
    - final_answer is set   → route to END
    - final_answer is None  → route back to sec_retriever for retry

Usage:
    from agents.graph import compiled_graph
    result = compiled_graph.invoke({"query": "...", "retry_count": 0})
    print(result["final_answer"])
"""

import logging
from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.nodes.query_analyzer import query_analyzer_node
from agents.nodes.sec_retriever import sec_retriever_node
from agents.nodes.market_analyst import market_analyst_node
from agents.nodes.critic import critic_node

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function — determines routing after Critic node runs.

    LangGraph calls this function after critic_node completes.
    Returns a string key that maps to the next node name.

    Logic:
        - final_answer is set → critic passed → route to END
        - final_answer is None → critic failed → route back to sec_retriever

    Args:
        state: Current AgentState after critic_node has updated it

    Returns:
        "end"   → routes to END node
        "retry" → routes back to sec_retriever node
    """
    if state.get("final_answer") is not None:
        logger.info(f"Critic PASSED (score={state.get('quality_score', 0):.2f}) → routing to END")
        return "end"
    else:
        logger.info(f"Critic RETRY (score={state.get('quality_score', 0):.2f}, "
                    f"retry_count={state.get('retry_count', 0)}) → routing to sec_retriever")
        return "retry"


# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Node registration:
        Each node is a Python function that accepts AgentState
        and returns a partial state update dict.

    Edge types:
        - Normal edge    : always transitions A → B
        - Conditional edge: calls a function to decide next node

    Returns:
        Compiled LangGraph app ready for .invoke()
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("sec_retriever",  sec_retriever_node)
    graph.add_node("market_analyst", market_analyst_node)
    graph.add_node("critic",         critic_node)

    # ── Normal edges (always execute in this order) ───────────────────────────
    graph.add_edge(START,            "query_analyzer")   # Entry point
    graph.add_edge("query_analyzer", "sec_retriever")    # After extraction → retrieve
    graph.add_edge("sec_retriever",  "market_analyst")   # After retrieval → synthesize
    graph.add_edge("market_analyst", "critic")           # After synthesis → evaluate

    # ── Conditional edge (retry loop) ─────────────────────────────────────────
    # After critic runs, should_retry() decides the next step:
    #   "end"   → graph terminates
    #   "retry" → loops back to sec_retriever for another retrieval attempt
    graph.add_conditional_edges(
        "critic",           # Source node
        should_retry,       # Routing function
        {
            "end":   END,              # PASS → terminate
            "retry": "sec_retriever",  # FAIL → retry retrieval
        }
    )

    return graph.compile()


# ── Compiled graph — import this in FastAPI and tests ────────────────────────
compiled_graph = build_graph()
