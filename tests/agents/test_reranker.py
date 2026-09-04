# tests/agents/test_reranker.py
#
# Tests `select_top_k_per_ticker` — the pure grouping/trimming function.
# No model load needed: the cross-encoder scores are passed in directly,
# same as `rerank_node` would produce them.

from agents.nodes.reranker import select_top_k_per_ticker


def _chunk(ticker, chunk_id):
    return {"chunk_id": chunk_id, "ticker": ticker, "text": "x", "vector_score": 0.5, "rerank_score": None}


def test_keeps_top_k_within_a_single_ticker():
    chunks = [_chunk("AAPL", f"AAPL:{i}") for i in range(8)]
    scores = [0.1, 0.9, 0.5, 0.7, 0.3, 0.95, 0.2, 0.6]

    kept = select_top_k_per_ticker(chunks, scores, k=3)

    assert len(kept) == 3
    assert [c["chunk_id"] for c in kept] == ["AAPL:5", "AAPL:1", "AAPL:3"]


def test_comparison_query_keeps_top_k_per_ticker_not_globally():
    # AMD chunks all score higher than AVGO's — a global top-k would starve AVGO.
    chunks = [_chunk("AMD", "AMD:0"), _chunk("AMD", "AMD:1"), _chunk("AVGO", "AVGO:0"), _chunk("AVGO", "AVGO:1")]
    scores = [0.9, 0.8, 0.4, 0.3]

    kept = select_top_k_per_ticker(chunks, scores, k=1)

    tickers_kept = {c["ticker"] for c in kept}
    assert tickers_kept == {"AMD", "AVGO"}
    assert len(kept) == 2


def test_empty_input_returns_empty():
    assert select_top_k_per_ticker([], [], k=5) == []


def test_rerank_score_is_attached_to_kept_chunks():
    chunks = [_chunk("AAPL", "AAPL:0")]
    scores = [0.77]

    kept = select_top_k_per_ticker(chunks, scores, k=5)

    assert kept[0]["rerank_score"] == 0.77
