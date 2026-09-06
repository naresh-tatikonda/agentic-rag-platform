# tests/agents/test_reranker.py
#
# Tests `select_top_k_per_ticker` — the pure grouping/trimming function —
# and the RERANK_ENABLED flag gate. No model load needed: the cross-encoder
# scores are passed in directly, same as `rerank_node` would produce them,
# and the default (flag off) path never touches sentence-transformers.

import pytest

from agents.nodes.reranker import select_top_k_per_ticker, rerank_enabled, rerank_node


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


# -- RERANK_ENABLED flag gate ------------------------------------------------

def test_rerank_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    assert rerank_enabled() is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", " On "])
def test_rerank_flag_truthy_values(monkeypatch, val):
    monkeypatch.setenv("RERANK_ENABLED", val)
    assert rerank_enabled() is True


def test_rerank_node_baseline_narrows_by_vector_score_without_loading_model(monkeypatch):
    # Flag off: must narrow purely on vector_score and never import the
    # cross-encoder. If it tried, this would raise (sentence-transformers
    # isn't a test dependency).
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    chunks = [
        {"chunk_id": "AAPL:0", "ticker": "AAPL", "text": "a", "vector_score": 0.2, "rerank_score": None},
        {"chunk_id": "AAPL:1", "ticker": "AAPL", "text": "b", "vector_score": 0.9, "rerank_score": None},
    ]
    out = rerank_node({"query": "q", "retrieved_chunks": chunks})
    assert [c["chunk_id"] for c in out["retrieved_chunks"]] == ["AAPL:1", "AAPL:0"]
    assert out["retrieval_scores"] == [0.9, 0.2]
