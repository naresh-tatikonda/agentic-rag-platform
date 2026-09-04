"""
agents/nodes/reranker.py
-------------------------
Reranker Node — third node in the LangGraph pipeline (route == "sec").

Responsibility:
    Re-scores the wide per-ticker candidate pool from SECRetriever with a
    cross-encoder, then keeps the top-k per ticker for synthesis.

Why a cross-encoder here, on top of vector search?
    The embedding model (bi-encoder) scores query and chunk independently,
    then compares vectors — cheap enough to search the whole corpus, but it
    has no token-level interaction between query and chunk, so it's easy to
    conflate "AAPL revenue 2023" with "AAPL revenue 2024" (same topic,
    wrong fact). A cross-encoder feeds [query, chunk] through the model
    together, so it catches exactly that kind of near-miss. It's too
    expensive to run over the whole corpus, which is why SECRetriever casts
    a wide net first and this node only re-scores that candidate set.

Why top-k PER TICKER, not top-k globally?
    A global top-k would let one ticker's chunks crowd out another's in a
    comparison query ("compare AMD and AVGO") if one company's filing text
    happens to score higher across the board. Reranking within each ticker
    guarantees every ticker gets a fair, non-empty context for synthesis.

Model:
    cross-encoder/ms-marco-MiniLM-L-6-v2 — small (~80MB), CPU-friendly,
    fast enough for the CANDIDATE_POOL_SIZE this reranks (see sec_retriever.py).
    Loaded once at module import (singleton) so it isn't reloaded per request.

Output:
    Updates AgentState with: retrieved_chunks (trimmed + rerank_score set),
    retrieval_scores (mirrors rerank_score)
"""

import logging

from agents.state import AgentState

logger = logging.getLogger(__name__)

RERANK_KEEP_PER_TICKER = 5
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model():
    """Lazy singleton — avoids importing/loading the model when reranker isn't used (e.g. in tests)."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def select_top_k_per_ticker(
    chunks: list[dict],
    scores: list[float],
    k: int = RERANK_KEEP_PER_TICKER,
) -> list[dict]:
    """
    Pure function — no model call. Takes chunks already annotated with a
    `rerank_score` (via `scores`, index-aligned with `chunks`) and returns
    the top-k per ticker, sorted descending by rerank_score.

    Separated from `rerank_node` so the grouping/trimming logic is
    unit-testable without loading the cross-encoder model.
    """
    by_ticker: dict[str, list[dict]] = {}
    for chunk, score in zip(chunks, scores):
        scored_chunk = {**chunk, "rerank_score": float(score)}
        by_ticker.setdefault(chunk["ticker"], []).append(scored_chunk)

    kept: list[dict] = []
    for ticker_chunks in by_ticker.values():
        ticker_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        kept.extend(ticker_chunks[:k])

    return kept


def rerank_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — cross-encoder rerank of the candidate pool.
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        logger.warning("Reranker received no candidate chunks — skipping")
        return {"retrieved_chunks": [], "retrieval_scores": []}

    try:
        model = _get_model()
        pairs = [(query, c["text"]) for c in chunks]
        raw_scores = model.predict(pairs)

        kept = select_top_k_per_ticker(chunks, raw_scores)
        logger.info(f"Reranker kept {len(kept)}/{len(chunks)} candidates across {len(set(c['ticker'] for c in chunks))} ticker(s)")

        return {
            "retrieved_chunks": kept,
            "retrieval_scores": [c["rerank_score"] for c in kept],
        }

    except Exception as e:
        # Fail open here, deliberately: a broken reranker shouldn't drop
        # retrieval to empty. Fall back to the vector-score ordering,
        # trimmed the same way, so downstream nodes still get a bounded
        # per-ticker context.
        logger.error(f"Reranker failed: {e}. Falling back to vector-score ranking.")
        vector_scores = [c["vector_score"] for c in chunks]
        kept = select_top_k_per_ticker(chunks, vector_scores)
        for c in kept:
            c["rerank_score"] = None
        return {
            "retrieved_chunks": kept,
            "retrieval_scores": [c["vector_score"] for c in kept],
        }
