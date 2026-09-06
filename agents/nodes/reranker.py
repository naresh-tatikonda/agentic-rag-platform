"""
agents/nodes/reranker.py
-------------------------
Candidate-narrowing node — fourth node in the pipeline (route == "sec").

Responsibility:
    Narrows SECRetriever's wide per-ticker candidate pool down to the top-k
    per ticker that MarketAnalyst will synthesize from.

    Two modes, controlled by the RERANK_ENABLED env flag:
      - OFF (default): narrow by the vector-similarity score SECRetriever
        already produced. This is the simple baseline — metadata filter
        (ticker + fiscal_year, in SQL) + dense retrieval, nothing learned
        on top.
      - ON: re-score every candidate with a cross-encoder that reads
        (query, chunk) jointly, then narrow by that score.

Why the flag / why default OFF:
    The candidate set is ALREADY scoped to the exact ticker and fiscal_year
    by SECRetriever's SQL WHERE clause, so the cross-encoder is NOT what
    prevents cross-filing/cross-year confusion — structured filtering does
    that, first and for free. The cross-encoder's only remaining job is
    fine-grained relevance ranking WITHIN one correctly-scoped filing
    (e.g. picking the specific liquidity-risk paragraph over generic
    risk-factor boilerplate). Whether that's worth a torch dependency and
    the added latency is an empirical question — the retrieval eval
    (recall@k / MRR) measures the delta with the flag off vs. on. Until
    that measurement justifies it, the baseline ships and the reranker is
    opt-in.

Why top-k PER TICKER, not top-k globally (applies in both modes):
    A global top-k would let one company's chunks crowd out another's in a
    comparison query ("compare AMD and AVGO") if one filing scores higher
    across the board. Narrowing within each ticker guarantees every ticker
    gets a fair, non-empty context.

Cross-encoder model (ON mode only):
    cross-encoder/ms-marco-MiniLM-L-6-v2 — small (~80MB), CPU-only, loaded
    once at first use (lazy singleton) so it isn't imported when the flag
    is off (e.g. in tests, or the default baseline).

Output:
    Updates AgentState with: retrieved_chunks (narrowed), retrieval_scores
"""

import logging
import os

from agents.state import AgentState

logger = logging.getLogger(__name__)

RERANK_KEEP_PER_TICKER = 5
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def rerank_enabled() -> bool:
    """Read the flag at call time (not import) so tests/eval can flip it per run."""
    return os.getenv("RERANK_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _get_model():
    """Lazy singleton — the cross-encoder is only imported/loaded when RERANK_ENABLED is on."""
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
    Pure function — no model call. Groups `chunks` by ticker, writes each
    chunk's `rerank_score` from `scores` (index-aligned), and returns the
    top-k per ticker sorted descending by that score.

    Used by both modes: the baseline passes vector-similarity scores here,
    the reranker passes cross-encoder scores.
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


def _narrow_by_vector_score(chunks: list[dict]) -> AgentState:
    kept = select_top_k_per_ticker(chunks, [c["vector_score"] for c in chunks])
    for c in kept:
        c["rerank_score"] = None
    logger.info(f"Baseline narrowing: kept {len(kept)}/{len(chunks)} by vector score")
    return {"retrieved_chunks": kept, "retrieval_scores": [c["vector_score"] for c in kept]}


def rerank_node(state: AgentState) -> AgentState:
    """LangGraph node — narrows the candidate pool, optionally via cross-encoder."""
    query = state["query"]
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        logger.warning("Reranker received no candidate chunks — skipping")
        return {"retrieved_chunks": [], "retrieval_scores": []}

    if not rerank_enabled():
        return _narrow_by_vector_score(chunks)

    try:
        model = _get_model()
        raw_scores = model.predict([(query, c["text"]) for c in chunks])
        kept = select_top_k_per_ticker(chunks, raw_scores)
        logger.info(f"Reranker kept {len(kept)}/{len(chunks)} across {len(set(c['ticker'] for c in chunks))} ticker(s)")
        return {"retrieved_chunks": kept, "retrieval_scores": [c["rerank_score"] for c in kept]}

    except Exception as e:
        # Fail open: a broken reranker falls back to the baseline narrowing
        # rather than dropping retrieval to empty.
        logger.error(f"Reranker failed: {e}. Falling back to vector-score narrowing.")
        return _narrow_by_vector_score(chunks)
