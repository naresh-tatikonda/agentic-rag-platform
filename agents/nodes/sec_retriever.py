"""
agents/nodes/sec_retriever.py
------------------------------
SECRetriever Node — second node in the LangGraph pipeline (route == "sec").

Responsibility:
    For each ticker in AgentState.tickers, embeds the query and runs pgvector
    HNSW cosine search scoped to that ticker + fiscal_year, returning a wide
    candidate set per ticker for the Reranker node to narrow down.

    Current retrieval is vector-only (cosine similarity via pgvector <=>).
    There is no lexical/BM25 stage yet — the GIN index in the schema exists
    but nothing here queries it. Hybrid search is tracked as future work,
    not implemented in this pass.

    Retrieving wide (CANDIDATE_POOL_SIZE) instead of a tight top-k matters
    because reranking can only reorder what's in the candidate set — it
    can't recover a chunk that vector search didn't surface at all.

Table schema (sec_filings):
    id, ticker, filing_type, filed_date, cik,
    chunk_index, chunk_text, embedding, created_at, fiscal_year

Output:
    Updates AgentState with: retrieved_chunks (list[dict], see agents/state.py),
    retrieval_scores (mirrors vector_score, pre-rerank)
"""

import logging
import os

import psycopg2
from openai import OpenAI

from agents.state import AgentState

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CANDIDATE_POOL_SIZE = 30      # per-ticker candidates handed to the Reranker
EMBEDDING_MODEL = "text-embedding-3-small"   # Must match ingestion model

# -- Intent -> keyword boost mapping ------------------------------------------
INTENT_KEYWORDS = {
    "risk_analysis": "risk factors threats litigation regulatory",
    "revenue_summary": "revenue earnings net income financial results",
    "business_overview": "business segments products services operations",
    "comparison": "",
    "general": "",
}


def get_db_connection():
    """Create a fresh PostgreSQL connection per node invocation."""
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def embed_query(text: str) -> list[float]:
    """Generate embedding vector for the query. Must match ingestion model (1536 dims)."""
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _search_ticker(cur, ticker: str, fiscal_year: int, embedding_str: str) -> list[dict]:
    """Run the per-ticker vector search and shape rows into the chunk-dict contract."""
    sql = """
        SELECT
            id,
            chunk_text,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM sec_filings
        WHERE ticker = %s AND fiscal_year = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    cur.execute(sql, (embedding_str, ticker, fiscal_year, embedding_str, CANDIDATE_POOL_SIZE))
    rows = cur.fetchall()

    if not rows:
        logger.warning(f"No chunks found for ticker={ticker}, fiscal_year={fiscal_year}")
        return []

    return [
        {
            "chunk_id": f"{ticker}:{row_id}",
            "ticker": ticker,
            "text": chunk_text,
            "vector_score": float(score),
            "rerank_score": None,
        }
        for row_id, chunk_text, score in rows
    ]


def sec_retriever_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — retrieves a wide per-ticker candidate pool from pgvector.

    Args:
        state: Current AgentState with tickers, fiscal_year, intent populated

    Returns:
        Partial AgentState update with retrieved_chunks and retrieval_scores
    """
    query = state["query"]
    tickers = state.get("tickers") or []
    fiscal_year = state.get("fiscal_year")
    intent = state.get("intent") or "general"

    if not tickers or fiscal_year is None:
        logger.error(f"SECRetriever: missing tickers or fiscal_year (tickers={tickers}, fiscal_year={fiscal_year})")
        return {"retrieved_chunks": [], "retrieval_scores": []}

    logger.info(f"SECRetriever searching tickers={tickers}, fiscal_year={fiscal_year}, intent={intent}")

    keyword_boost = INTENT_KEYWORDS.get(intent, "")
    enriched_query = f"{query} {keyword_boost}".strip()

    try:
        query_embedding = embed_query(enriched_query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        conn = get_db_connection()
        cur = conn.cursor()

        all_chunks: list[dict] = []
        for ticker in tickers:
            all_chunks.extend(_search_ticker(cur, ticker, fiscal_year, embedding_str))

        cur.close()
        conn.close()

        if all_chunks:
            logger.info(f"SECRetriever retrieved {len(all_chunks)} candidate chunks across {len(tickers)} ticker(s)")

        return {
            "retrieved_chunks": all_chunks,
            "retrieval_scores": [c["vector_score"] for c in all_chunks],
        }

    except Exception as e:
        logger.error(f"SECRetriever failed: {e}")
        return {"retrieved_chunks": [], "retrieval_scores": []}
