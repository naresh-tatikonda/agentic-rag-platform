"""
agents/nodes/market_analyst.py
-------------------------------
MarketAnalyst Node — third node in the LangGraph pipeline.

Responsibility:
    Takes retrieved SEC filing chunks and synthesizes a comprehensive,
    structured answer using GPT-4o.

Why GPT-4o here (not gpt-4o-mini)?
    - QueryAnalyzer: extraction task → gpt-4o-mini (cheap, fast)
    - MarketAnalyst: synthesis task → gpt-4o (better reasoning, richer answers)
    - Cost routing: use the right model for the right task

Output:
    Updates AgentState with: draft_answer
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AgentState

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── LLM client — GPT-4o for high-quality synthesis ───────────────────────────
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,      # Slight creativity for natural language, mostly deterministic
    max_tokens=1000,      # Enough for a thorough but concise answer
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior financial analyst specializing in SEC filing analysis.

Your job is to answer questions about companies based ONLY on the provided SEC filing excerpts.

Guidelines:
- Base your answer strictly on the provided context — do not hallucinate facts
- Be specific — cite numbers, percentages, and dates when available in the context
- Structure your answer clearly with key points
- If the context doesn't contain enough information, say so explicitly
- Keep the answer focused and under 300 words
"""


def format_chunks(chunks: list[str], scores: list[float]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.
    Including similarity scores helps the LLM weight higher-quality chunks.

    Args:
        chunks: List of text chunks from pgvector
        scores: Corresponding similarity scores

    Returns:
        Formatted string of numbered excerpts with scores
    """
    formatted = []
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        formatted.append(
            f"[Excerpt {i} | Relevance: {score:.3f}]\n{chunk.strip()}"
        )
    return "\n\n".join(formatted)


def market_analyst_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — synthesizes retrieved chunks into a draft answer.

    Args:
        state: Current AgentState with retrieved_chunks and retrieval_scores populated

    Returns:
        Partial AgentState update with draft_answer populated
    """
    query       = state["query"]
    chunks      = state.get("retrieved_chunks") or []
    scores      = state.get("retrieval_scores") or []
    ticker      = state.get("ticker")
    fiscal_year = state.get("fiscal_year")

    if not ticker or not fiscal_year:
        logger.error("MarketAnalyst: ticker or fiscal_year missing from state")
        return {"final_answer": "Error: query context missing ticker or fiscal year.", "quality_score": 0.0}

    logger.info(f"MarketAnalyst synthesizing answer from {len(chunks)} chunks")

    # ── Handle empty retrieval gracefully ─────────────────────────────────────
    if not chunks:
        logger.warning("MarketAnalyst received no chunks — returning fallback answer")
        return {
            "draft_answer": (
                f"I could not find relevant information about {ticker} "
                f"in the FY{fiscal_year} SEC filing to answer your question."
            )
        }

    # ── Format context from retrieved chunks ──────────────────────────────────
    context = format_chunks(chunks, scores)

    # ── Build user prompt with correct fiscal year label ─────────────────────
    # fiscal_year reflects the year the 10-K COVERS, not the download/storage date
    user_prompt = f"""Question: {query}

SEC Filing Context ({ticker}, FY{fiscal_year} 10-K):
{context}

Please provide a comprehensive answer based on the above SEC filing excerpts."""

    try:
        # ── Call GPT-4o for synthesis ─────────────────────────────────────────
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        draft_answer = response.content.strip()
        logger.info(f"MarketAnalyst generated answer ({len(draft_answer)} chars)")

        return {"draft_answer": draft_answer}

    except Exception as e:
        logger.error(f"MarketAnalyst failed: {e}")
        return {
            "draft_answer": f"Analysis failed due to an error: {str(e)}"
        }
