"""
agents/nodes/market_analyst.py
-------------------------------
MarketAnalyst Node — fourth node in the LangGraph pipeline (route == "sec").

Responsibility:
    Synthesizes the reranked SEC filing chunks into a set of discrete claims,
    each tagged with the chunk it came from — not a free-text paragraph.

Why claims instead of free text?
    Citations on a free-text answer only prove the model attached a source,
    not that the source actually says what the sentence claims. Forcing the
    model to emit {text, source_chunk_id} pairs via structured output makes
    every factual statement point at a specific, checkable piece of context,
    and gives the Critic node something concrete to verify per-claim instead
    of judging the answer holistically.

Why GPT-4o here (not gpt-4o-mini)?
    QueryAnalyzer/Critic are extraction/scoring tasks -> gpt-4o-mini (cheap).
    MarketAnalyst is synthesis -> gpt-4o (better reasoning over multi-chunk,
    multi-ticker context). Cost routing: right model for the right task.

Output:
    Updates AgentState with: draft_claims, draft_answer (claims composed to prose)
"""

import logging
import os

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from agents.state import AgentState

logger = logging.getLogger(__name__)

client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


class Claim(BaseModel):
    text: str = Field(description="A single factual statement, standalone and specific")
    source_chunk_id: str = Field(description="chunk_id of the retrieved excerpt that supports this claim")


class DraftClaims(BaseModel):
    claims: list[Claim] = Field(description="The answer decomposed into individually-sourced claims")


SYSTEM_PROMPT = """You are a senior financial analyst specializing in SEC filing analysis.

Answer the question using ONLY the provided, numbered SEC filing excerpts. Decompose your answer
into individual claims. Every claim MUST cite the chunk_id of the excerpt it came from.

Rules:
- Do not state anything the excerpts don't support — no outside knowledge, no estimates, no filling gaps.
- Every number, percentage, or date in a claim must appear in its cited excerpt.
- If the excerpts don't fully answer the question, say so as a claim citing the closest relevant excerpt
  rather than inventing the missing part.
- When multiple tickers are present, produce claims for each and make the comparison explicit
  (e.g. "AMD's R&D spend was X" and "AVGO's R&D spend was Y" as separate claims).
"""


def format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks into a chunk_id-addressable context block."""
    formatted = []
    for c in chunks:
        score = c.get("rerank_score") if c.get("rerank_score") is not None else c.get("vector_score")
        formatted.append(f"[chunk_id: {c['chunk_id']} | ticker: {c['ticker']} | relevance: {score:.3f}]\n{c['text'].strip()}")
    return "\n\n".join(formatted)


def compose_answer(claims: list[Claim]) -> str:
    """Join claims into prose for display / RAGAS. The claims themselves remain the source of truth."""
    return " ".join(c.text for c in claims)


def market_analyst_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — synthesizes reranked chunks into sourced claims.
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks") or []
    tickers = state.get("tickers") or []
    fiscal_year = state.get("fiscal_year")

    logger.info(f"MarketAnalyst synthesizing from {len(chunks)} chunks for tickers={tickers}")

    if not chunks:
        logger.warning("MarketAnalyst received no chunks — returning fallback answer")
        fallback = f"I could not find relevant information about {', '.join(tickers)} in the FY{fiscal_year} SEC filing(s) to answer your question."
        return {"draft_claims": [], "draft_answer": fallback}

    context = format_chunks(chunks)
    user_prompt = f"""Question: {query}

SEC Filing Excerpts (FY{fiscal_year}):
{context}

Decompose your answer into claims, each citing its source chunk_id."""

    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            max_retries=1,
            response_model=DraftClaims,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        claims = [c.model_dump() for c in result.claims]
        draft_answer = compose_answer(result.claims)
        logger.info(f"MarketAnalyst generated {len(claims)} claims")

        return {"draft_claims": claims, "draft_answer": draft_answer}

    except Exception as e:
        logger.error(f"MarketAnalyst failed: {e}")
        return {"draft_claims": [], "draft_answer": f"Analysis failed due to an error: {e}"}
